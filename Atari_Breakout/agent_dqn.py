#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from collections import deque
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.replay_buffer_size = 10000
        self.start_to_learn = 5000
        self.update_target_net = 5000
        self.learning_rate = 1.5e-4
        self.batch_size = 32
        self.buffer_ = deque(maxlen=self.replay_buffer_size)
        self.epsilon = 0.99
        self.epsilon_ = 0.2
        
        
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.Net = DQN()

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, epsilon, device='cpu', test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            observation = torch.tensor(observation).to(device)
            q_values = self.Net(observation)
            _,action_value = torch.max(q_values,dim=-1)
            action = int(action_value.item())
        ###########################
        return action
    
    def push(self, sample):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.buffer_.append(sample)
        ###########################
        
        
    def replay_buffer(self,batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        idx_s = random.sample(range(len(self.buffer_)),batch_size)
        minibatch = []
        for _,i in enumerate(idx_s):
            minibatch.append(i)
        ###########################
        return np.array(minibatch)
    
    def get_reward(self,action):
        next_state,reward,terminal,_ = env.step(action)
        return next_state,reward,terminal

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        optimizer = optim.Adam(net.parameters(),lr=self.learning_rate)
        total_r = []
        best_r = None
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        while True:
            observation = env.reset()
            sum_reward = 0
            i = 0
            self.epsilon = max(epsilon_,self.epsilon-0.1/i)
            while True:
                action_training = self.make_action(observation,self.epsilon,device)
                next_state,reward,terminal = get_reward(action_training)
                sum_reward += reward
                if terminal:
                    break
                else:
                    observation = next_state
            total_r.append(sum_reward)
            if total_r is not None:
                mean_reward = np.mean(total_r)
                if mean_reward > 40:
                    print("DOne")
                    break
                
        ###########################
