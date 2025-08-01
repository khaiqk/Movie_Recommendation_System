import random 
import numpy as np
import pandas as pd

import os
import sys
from dotenv import load_dotenv

load_dotenv()

project_root = os.getenv("ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from processed_data.users_data import data_users
from processed_data.movies_data import data_movies
from DQN.model import DQL_model

class Env:
    def __init__(self, data_movies: pd.DataFrame = data_movies, data_users: pd.DataFrame= data_users, user_id: str=0, model: DQL_model = None):
        # load data users and movies
        self.data_users = data_users
        self.user_id = user_id
        self.data_movies = data_movies
        self.model = model  # dqn model
        
        # get user
        self.user_data = self.data_users[self.data_users['userId'] == self.user_id]
        self.user_data.drop(columns=['userId'], inplace=True)
        
        # get genres
        self.genres = self.user_data.columns[1:-1]
        self.user_data.drop(columns=['movieId'], inplace=True)
        
        # storage data
        self.storage = self.user_data[self.genres]
        self.memory = None
        self.movie_suggestions = None
                
        # rating movie
        self.refresh_count = 0
        self.max_refresh = 10
        
        self.total_reward = [0]
        self.list_user_satisfaction = []
        
        self.get_memory_from_user_data() 
        
        self.model.user_satisfaction, self.model.stability = self.calc_user_statisfaction()    
        
        self.reset()
        
    def reset(self): 
        self.refresh_count = 0
        self.movie_suggestions = self.data_movies.sample(10)
        print('Reset')
        print(self.movie_suggestions['title'].values)
        return self.movie_suggestions
    
    def suggest_next_movie(self):
        if self.refresh_count == self.max_refresh or self.memory is None:
            return self.reset()
        else:    
            self.movie_suggestions = None
            
            current_state = self.memory[self.genres].values.astype(np.float32)
                
            suggested_genres_indices = self.model.act(current_state)
                            
            for genre_index in suggested_genres_indices:
                filtered_movie = self.data_movies[self.data_movies[self.genres[genre_index]] == 1]
                
                if not filtered_movie.empty:
                    print(f"genre: {self.genres[genre_index]}")
                    self.movie_suggestions = pd.concat([self.movie_suggestions, filtered_movie.sample(1)])
                else:
                    self.movie_suggestions = pd.concat([self.movie_suggestions, self.data_movies.sample(1)])
            
            self.movie_suggestions = pd.concat([self.movie_suggestions, self.data_movies.sample(10 - len(suggested_genres_indices))])
            
            print(self.movie_suggestions['title'].values)
            return self.movie_suggestions
        
    
    def storage_data(self):
        if self.memory is not None:
            new_row = pd.DataFrame([self.memory[self.genres].values], columns=self.genres)
            self.storage = pd.concat([self.storage, new_row], ignore_index=True)
        return self.storage
    
    def reward_movie(self, rating):
        reward = rating
        
        # refresh
        # penalty_refresh =  self.refresh_count * 0.04
        # reward -= penalty_refresh        

        # user satisfaction
        if self.model.user_satisfaction >= 4:
            reward += rating * 0.03
        elif self.model.user_satisfaction < 2:
            reward -= rating * 0.08
        
        reward = np.clip(reward, 0, 5)
        
        self.model.user_satisfaction, self.model.stability = self.calc_user_statisfaction()
        
        print(f"user satisfaction: {self.model.user_satisfaction}")
        print(f"stability: {self.model.stability}")
        
            
        return reward 
    
    def rating_movie(self, rating= None):
        if rating is not None and self.memory is not None:
            
            self.memory = self.memory[self.genres]
            
            state = self.memory.values.astype(np.float32)
            
            action = self.model.act(state)
            
            reward = self.reward_movie(rating)
            
            self.total_reward.append(reward + self.total_reward[-1])
            
            next_state = state
            
            self.model.remember(state, action, reward, next_state)
            self.model.train()
            
            self.refresh_count = 0
            
            return self.memory
    
    def step(self, index = None, refresh= False):
        
        if refresh== True:
            self.refresh_count += 1
            if self.refresh_count == self.max_refresh:
                self.reset()  
            else:
                if self.model.memory and len(self.model.memory) > 0:
                    state, action, reward, next_state = self.model.memory[-1]
                    reward -= 0.04
                    
                    self.total_reward.append(reward + self.total_reward[-1])
                    
                    self.model.user_satisfaction, self.model.stability = self.calc_user_statisfaction()
                    
                    self.model.remember(state, action, reward, next_state) 
                    
                    self.model.train()
                     
        if index is not None:
                                    
            self.selected_movie = self.movie_suggestions.iloc[index]
            
            self.memory = self.selected_movie
            
            return self.selected_movie
        
    def get_memory_from_user_data(self):
        for index, row in enumerate(self.user_data.itertuples()):
            state = self.user_data.iloc[index][self.genres].values
            
            action = [index for index, value in enumerate(state) if value == 1]
            
            if action:
                action = random.choice(action)
                            
            rating = row.rating
            
            reward = self.reward_movie(rating)
            
            self.total_reward.append(reward + self.total_reward[-1])
            
            if index + 1 < len(self.user_data):
                next_state = self.user_data.iloc[index + 1][self.genres].values
            else:
                next_state = self.user_data.iloc[index][self.genres].values
            
            self.model.remember(state, action, reward, next_state)
            
        self.model.train()
    
    def calc_user_statisfaction(self):
        list_rewards = [self.total_reward[i] - self.total_reward[i-1] for i in range(1, len(self.total_reward))]
        
        user_satisfaction = np.mean(list_rewards)
        stability = np.std(list_rewards)
                
        self.list_user_satisfaction.append(user_satisfaction)

        return user_satisfaction, stability
