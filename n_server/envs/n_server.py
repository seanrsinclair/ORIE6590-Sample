import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class NServerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        
        # Initializes model parameters based on a configuration dictionary
        self.rho = config['rho']
        self.N = config['N']
        self.h1 = config['h1']
        self.h2 = config['h2']
        self.starting_state = config['starting_state']

        # Updates arrival rates

        self.lambda1 = 1.3*self.rho
        self.lambda2 = 0.4*self.rho

        # Defines service rates
        self.mu1 = 1
        self.mu2 = 0.5
        self.mu3 = 1


        # Normalization factor for the uniformization
        self.B = self.lambda1 + self.lambda2 + self.mu1 + self.mu2 + self.mu3

        # Defines state and action spaces, sets current state to be starting_state
        self.action_space = gym.spaces.MultiDiscrete([2,4])
        self.observation_space = gym.spaces.MultiDiscrete([self.N+1,self.N+1])
        self.state = np.asarray(self.starting_state)

    # Resets environment to initial state
    def reset(self):
        self.state = np.asarray(self.starting_state)
        return self.state

    # Defines one step of the MDP, returning the new state, reward, whether time horizon is finished, and a dictionary of information
    def step(self, action):
        reward = self.r(self.state)
        
        # Just for personal double checking for Q learning algortihm, can ignore this
        # if self.state[0] == self.N and self.state[1] == self.N and action[0] == 0 and action[1] == 0:
            # print('Uh oh, stuck at absorbing state')
        
        
        trans = self.pr(self.state, action)
        states = list(trans.keys())
        probs = list(trans.values())
        # Computes new state
        newState = states[np.random.choice(range(len(states)), 1, p = probs)[0]]
        self.state = np.asarray(newState)
        episode_over = False

        return self.state, reward, episode_over, {}



    # Auxilary function computing the reward
    def r(self, state):
        return -(self.h1*state[0] + self.h2*state[1])/self.B

    # Auxilary function computing transition distribution
    def pr(self, state, action):
        transition_probs = {}
        if state[0] < self.N:
            transition_probs[(state[0]+1, state[1])] = self.lambda1/self.B
        
        if state[1] < self.N:
            transition_probs[(state[0], state[1]+1)] = self.lambda2/self.B

        if state[0] > 0:
            if action[0] == 1 and action[1] == 2:
                transition_probs[(state[0]-1, state[1])] = (self.mu1+self.mu2)/self.B
            elif action[0] == 1:
                transition_probs[(state[0]-1, state[1])] = (self.mu1)/self.B
            elif action[1] == 2:
                transition_probs[(state[0]-1, state[1])] = (self.mu2)/self.B

                
        if state[1] > 0:
            if action[1] == 3:
                transition_probs[(state[0], state[1]-1)] = (self.mu3)/self.B

        transition_probs[(state[0], state[1])] = 1 - sum(transition_probs.values())
        
        return transition_probs
