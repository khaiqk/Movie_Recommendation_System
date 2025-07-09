import pandas as pd
import keras
import numpy as np
import tensorflow as tf

class REINFORCE:
    def __init__(self, state_size: int = 19, action_size: int =19, learning_rate: float=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
        
        self.user_satisfaction, self.stability = 0, 0
        
        self.states = []
        self.actions = []
        self.rewards = []
        
    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy')
        return model
    
    def act(self, state):
        num_rec = 0
        if self.user_satisfaction > 4:
            num_rec = 6
        elif 3 <= self.user_satisfaction <= 4:
            num_rec = 5
        else:
            num_rec = 4
            
        state = state.reshape(1, -1)
        probabilities = self.model.predict(state, verbose=0)[0]
        actions = np.random.choice(self.action_size, num_rec, p=probabilities, replace=False)
        
        print(f"top {num_rec} indices: {actions}")
        return actions
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def update_policy(self):
        if not self.states:
            return
        
        discounted_rewards = self._discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-7)

        actions_one_hot = np.zeros((len(self.actions), self.action_size))
        for idx, action in enumerate(self.actions):
            actions_one_hot[idx, action] = 1
        
        with tf.GradientTape() as tape:
            logits = self.model(np.array(self.states), training=True)
            log_prob = tf.math.log(tf.reduce_sum(logits * actions_one_hot, axis=-1))
            
            loss = -tf.reduce_mean(log_prob * discounted_rewards)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    
    def _discount_rewards(self, rewards, gamma=0.99):
        # G_t = R_t + γR_{t+1} + γ^2R_{t+2} + ...
        G = np.zeros_like(rewards, dtype=np.float32)
        cumulative_sum = 0
        for t in reversed(range(len(rewards))):
            cumulative_sum = cumulative_sum * gamma + rewards[t]
            G[t] = cumulative_sum
        return G