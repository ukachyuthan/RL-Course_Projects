#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Initialize a deep Q-learning network
    
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    
    This is just a hint. You can build your own structure.
    """
    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.conv1 = nn.Conv2d(4,32,8,4)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.fc1_advantage = nn.Linear(7*7*64,512)
        self.fc1_scalar = nn.Linear(7*7*64,512)
        self.fc2_advantage = nn.Linear(512,4)
        self.fc2_scalar = nn.Linear(512,1)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        advantage = self.relu(self.fc1_advantage(x))
        scalar = self.relu(self.fc1_scalar(x))
        advantage = self.fc2_advantage(advantage)
        scalar = self.fc2_scalar(scalar).expand(x.size(0), 4)
        x = scalar+advantage-advantage.mean(1).unsqueeze(1).expand(x.size(0), 4)
        ###########################
        return x
