#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v mc_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise
    
    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    player_score = observation[0]
    # action
    if player_score >= 20:
        action = 0
    else:
        action = 1
    ############################
    return action 

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function 
        by using Monte Carlo first visit algorithm.
    
    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for i in range (n_episodes):
        print(i)
        # initialize the episode
        episode = []
        # generate empty episode
        state = env.reset()
        # loop until episode generation is done
        flag = 1
        new_flag = 0
        visited_state = {}
        while flag == 1:            
            # select an action
            action = policy(state)
            if new_flag == 0:
                visited_state = {state:new_flag}
            else:
                if state not in visited_state:
                    temp_dict = {state:new_flag}
                    visited_state.update(temp_dict)
            # return a reward and new state
            next_state,reward,done,info = env.step(action)
            # append state, action, reward to episode
            episode.append((state,action,reward))
            # update state to new state
            state = next_state
            new_flag = new_flag+1
            if (done):
                break
        for unique_state in visited_state:
            G = 0
            for j in range(visited_state[unique_state],len(episode)):
                G= G*gamma + episode[j][2]
            returns_sum[unique_state] += G
            returns_count[unique_state] += 1
            V[unique_state] = returns_sum[unique_state]/returns_count[unique_state]
            
        # get unique states set
        # each state should be a tuple so that we can use it as a dict key
    ############################
    
    return V

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    prob = random.uniform(0,1)
    if prob <= 1-epsilon:
        action = np.argmax(Q[state])
    else:
        action = random.randint(0,nA-1)
    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts. 
        Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = 1-0.1/n_episode during each episode
    and episode must > 0.    
    """
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    ############################
    # YOUR IMPLEMENTATION HERE #
    epsilon = 0.5
    
    for i in range (n_episodes+1,1,-1):
        epsilon = epsilon-0.1/i
        episode = []
        state = env.reset()
        visited_state = defaultdict(float)
        new_flag = 0
        while True:
            action = epsilon_greedy(Q,state,env.action_space.n,epsilon)
            if (state,action) not in visited_state:
                visited_state[(state,action)] = new_flag
            next_state,reward,done,info = env.step(action)
            episode.append((state,action,reward))
            state = next_state
            new_flag += 1
            if done:
                break

        for unique_state in visited_state:
            G = 0
            for j in range(visited_state[unique_state],len(episode)):
                G= G*gamma + episode[j][2]
            returns_sum[unique_state] += G
            returns_count[unique_state] += 1
            Q[unique_state[0]][unique_state[1]] = returns_sum[unique_state]/returns_count[unique_state]
            
        
    return Q