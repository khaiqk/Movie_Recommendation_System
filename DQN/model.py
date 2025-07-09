import numpy as np
import random
import keras
from collections import deque

class DQL_model:
    def __init__(self, state_size: int = 19, action_size: int = 19, 
            epsilon: float = 0.6, gamma: float = 0.75 , epsilon_decay: float = 0.99, 
            learning_rate: float = 0.01, batch_size: int = 32, tau: float = 0.05):
        
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tau = tau
        self.user_satisfaction, self.stability = 0, 0
        
        self.memory = deque(maxlen=2000)
                
        self.model = self.build_model()
        self.target_model = self.build_model()
        
    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(64, input_dim= self.state_size, activation= 'relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(32, activation= 'relu'))
        model.add(keras.layers.Dense(self.action_size))
        
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate= self.learning_rate))
        
        return model
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def act(self, state):
        num_rec = 0
        if self.user_satisfaction > 4:
            num_rec = 6
        elif 3 <= self.user_satisfaction <= 4:
            num_rec = 5
        else:
            num_rec = 4
            
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size, size= num_rec, replace=False)
        
        state = state.astype(np.float32)
        act_values = self.model.predict(np.reshape(state, [1, self.state_size]))
        
        actions = np.argsort(act_values[0])[-num_rec:][::-1]
        
        print(f"top {num_rec} indices: {actions}")
        return actions
    
    def update_target_model(self):
        model_w =  self.model.get_weights()
        target_model_w = self.target_model.get_weights()
        
        target_model_w = [self.tau * model_w[i] + (1 - self.tau) * target_model_w[i] for i in range(len(target_model_w))]  
        
        self.target_model.set_weights(target_model_w)
    
    def adjust_parameters(self):
        if self.user_satisfaction < 3:
            self.epsilon = min(0.9, self.epsilon + 0.05)
        if self.user_satisfaction > 3:
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            
        if self.stability < 0.4:
            self.gamma = max(0.9, self.gamma * 1.05)
        if self.stability > 1:
            self.gamma = min(0.5, self.gamma - 0.05)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
                
        minibatch = random.sample(self.memory, self.batch_size)
                
        for state, action, reward, next_state in minibatch:

            current_q_values = self.model.predict(np.reshape(state, [1, self.state_size]))

            next_q_values = self.target_model.predict(np.reshape(next_state, [1, self.state_size]))

            # Bellman
            target_q_value = reward + self.gamma * np.amax(next_q_values[0])

            target_f = current_q_values.copy()
            target_f[0][action] = target_q_value
            
            self.model.fit(np.reshape(state, [1, self.state_size]), target_f, epochs=1, verbose=0)      
        
        self.update_target_model()
                    
        self.adjust_parameters()