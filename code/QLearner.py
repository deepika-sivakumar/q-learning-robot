"""                     
Q Learner
"""

import numpy as np
import random as rand

class QLearner(object):

    def author(self):
        return 'Deepika'

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.num_states = num_states # No. of States
        self.num_actions = num_actions # No. of Actions
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discounted rate
        self.rar = rar # Random action rate
        self.radr = radr #Random action decay rate
        self.dyna = dyna # No of dyna updates
        self.verbose = verbose

        # Create the Q-table as a 2d-numpy array initialized to zeroes
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)

        # Initialize state & action
        self.s = 0
        self.a = 0

        # Dyna - Q Model
        # Initialize Transition Cost matrix with a very small number
        self.Tc = np.empty((num_states,num_actions,num_states),dtype=float)
        self.Tc.fill(0.00001)
        # Initialize Transition matrix
        self.T = np.zeros((num_states,num_actions,num_states),dtype=float)
        # Initialize Expected Reward matrix
        self.R = np.zeros((num_states,num_actions),dtype=float)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        # Set the state
        self.s = s
        # Roll the dice (Generate a Random number)
        random_num = rand.random()
        # If the random number value is less than random action rate, choose action randomly
        if(random_num < self.rar):
            action = rand.randint(0, self.num_actions-1)
        # Else, choose the action from the Q-table which has the maximum value for the state
        else:
            # Get the index of the action, which has the max value for the row state s_prime
            action = np.argmax(self.q_table[s, :])

        # Preserve the action for next time
        self.a = action

        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The immediate reward for state s & action a
        @returns: The selected action
        """
        # Get the best action for state s_prime
        max_action_prime = np.argmax(self.q_table[s_prime, :])
        # Update the Q-table with the reward r for the last state s & action a
        # Q_prime[s,a] = (1-alpha)*Q[s,a] + alpha*(immed_reward + gamma* Q[s_prime, argmax(Q[s_prime,a_prime])])
        self.q_table[self.s, self.a] = (1-self.alpha) * self.q_table[self.s,self.a] + self.alpha * (r + self.gamma * self.q_table[s_prime, max_action_prime])

        # Roll the dice (Generate a Random number)
        random_num = rand.random()
        # If the random number value is less than random action rate, choose action randomly
        if(random_num < self.rar):
            action = rand.randint(0, self.num_actions-1)
        # Else, choose the action from the Q-table which has the maximum value for the state
        else:
            # Get the index of the action, which has the max value for the row state s_prime
            action = np.argmax(self.q_table[s_prime, :])
        # Update the random action rate = random action rate * random action decay rate
        self.rar = self.rar * self.radr

        # Use Dyna - Q model only when dyna is greater than 0
        if(self.dyna > 0):
            # Dyna Q - Update Model
            self.dynaQ_update_model(s_prime,r)
            # Hallucinate and update the Q-table given dyna times
            for i in range(self.dyna):
                self.dynaQ_hallucinate()

        # Preserve the new states for next time
        self.s = s_prime
        self.a = action

        if self.verbose: print "s =", s_prime,"a =",action,"query r =",r
        return action

    def dynaQ_update_model(self,s_prime,r):
        # Increment Transition count
        self.Tc[self.s,self.a,s_prime] = self.Tc[self.s,self.a,s_prime] + 1
        # Update Transition matrix
        self.T[self.s,self.a,s_prime] = self.Tc[self.s,self.a,s_prime] / np.sum(self.Tc[self.s,self.a,:])
        # Update expected reward matrix
        self.R[self.s,self.a] = ((1-self.alpha) * self.R[self.s,self.a]) + (self.alpha * r)

    def dynaQ_hallucinate(self):
        # Pick a random state
        s = rand.randint(0, self.num_states-1)
        # Pick a random action
        a = rand.randint(0, self.num_actions-1)
        # Infer the s_prime by choosing the one which has the max probability of occurring for given s & a from the Transition matrix 
        s_prime = np.argmax(self.T[s,a,:])
        # Get the expected reward
        r = self.R[s,a]
        # Update the Q table with these values
        max_action_prime = np.argmax(self.q_table[s_prime, :])
        self.q_table[s,a] = (1-self.alpha) * self.q_table[s,a] + self.alpha * (r + self.gamma * self.q_table[s_prime, max_action_prime])

if __name__=="__main__":
    print "Q Learner"
