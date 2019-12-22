### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
           
    ############################
    # YOUR IMPLEMENTATION HERE #
    k = 1
    i=0
    while k==1:
        delta = 0
        i=i+1
	# For nS states, obtaining the value function
        for s in range (nS):
            v = value_function[s]
            v_new=0
	    # Enumerating Policy function of all states to calculate new Value function
            for a,ap in enumerate (policy[s]):
                for p,n_s,r,d in P[s][a]:
                    v_new += p*ap*(r+gamma*value_function[n_s])
            value_function[s] = v_new
            delta = max(delta,abs(v-value_function[s]))
        # Stopping iteration when no considerable change in value function from previous timestep 
        if ((tol>delta)):
            k=0
            break
    ############################
    return np.array(value_function)


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """
    # Initializing new policies as all zeros as new policy will be 1 for action and 0 otherwise
    new_policy = np.zeros([nS,nA])
	############################
	# YOUR IMPLEMENTATION HERE #
    # For nS states generate the ideal action to be taken
    for s in range (nS):
        quality_function = np.zeros(nA)
	# Find the q value for each action a in state s
        for a in range (nA):
            for p,n_s,r,d in P[s][a]:
                quality_function[a] += p*(r+gamma*value_from_policy[n_s])
        # Assign the argument of maximum q-value with 1, rest all 0s
        new_policy[s][np.argmax(quality_function)] = 1
	############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #
    max_iter = pow(nA,nS)
    # Run for a maximum of nA^nS iterations
    for i in range (max_iter):
        old_policy = new_policy.copy()
	# Get value function from policy
        value_function = policy_evaluation(P,nS,nA,new_policy)
	# improve the policy from new value function
        new_policy = policy_improvement(P,nS,nA,value_function)
	# return new policy and value function if no change in policy occurs
        if np.array_equal(old_policy,new_policy):
            return new_policy,value_function
    ############################
    

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    
    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #
    k = 1
    while k==1:
        delta = 0
	# iterate for nS states
        for s in range (nS):
            v = V_new[s]
            quality_function = np.zeros(nA)
	    # Find the quality function for each action for each state
            for a in range (nA):
                for p,n_s,r,d in P[s][a]:
                    quality_function[a] += p*(r+gamma*V_new[n_s])
	    # Assign the maximum of q_value to new Value function
            V_new[s] = np.max(quality_function)
            delta = max(delta,np.abs(v-V_new[s]))
	# Break if no change in value
        if delta<tol:
            break
    # Obtain new policy based on new optimal value function
    policy_new = policy_improvement(P,nS,nA,V_new)
    ############################
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #
	    # Get the action to be taken at state ob based on policy
            action = np.argmax(policy[ob])
	    # get next state, rewards,terminal and probability information for
	    # each step based on action
            n_s,r,d,prob = env.step(action)
	    # Calculate total rewards
            total_rewards += r
	    # change ob to new current state
            ob = n_s
	    # Break if terminal state
            if d:
                break
    return total_rewards
