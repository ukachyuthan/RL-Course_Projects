#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

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
        You can use the function from project2-1
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

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for i in range (n_episodes+1,1,-1):
        # define decaying epsilon
        epsilon = 0.99*epsilon

        # initialize the environment 
        state = env.reset()
        
        # get an action from policy
        action = epsilon_greedy(Q,state,env.action_space.n,epsilon)
        # one step in the environment
        episode = []
        while True:
            if state not in Q:
                Q[state]
            next_state, reward, done, info = env.step(action)
            # return a new state, reward and done
            episode.append((state,action,reward,done))
            # get next action
            next_action = epsilon_greedy(Q,next_state,env.action_space.n,epsilon)
            if next_state not in Q:
                Q[next_state]
            # TD update
            # td_target
            Q[state][action] = Q[state][action] + alpha*(reward+gamma*Q[next_state][next_action] - Q[state][action])
            # td_error

            # new Q

            
            # update state
            state = next_state
            # update action
            action = next_action
            if done:
                break
    ############################
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    # loop n_episodes
    for i in range (n_episodes+1,1,-1):
        
        # initialize the environment 
        #epsilon = 0.99*epsilon
        state = env.reset()
        # one step in the environment
        while True:
            # get an action from policy
            if state not in Q:
                Q[state]
            action = epsilon_greedy(Q,state,env.action_space.n,epsilon)
            # return a new state, reward and done
            next_state,reward,done,info = env.step(action)
            if next_state not in Q:
                Q[next_state]
            # TD update
            # td_target with best Q
            Q[state][action] = Q[state][action] + alpha*(reward+gamma*np.max(Q[next_state])-Q[state][action])
            # td_error
            print("done",done,"action",action)
            # new Q
            
            # update state
            state = next_state
            if done:
                break
    ############################
    return Q