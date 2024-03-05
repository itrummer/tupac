'''
Created on Mar 5, 2024

@author: immanueltrummer
'''
from gymnasium import Env
from gymnasium.spaces import Discrete


class PgSimEnv(Env):
    """ Environment for Postgres tuning (simulated). """
    
    def __init__(self):
        """ Initializes database for given credentials. """
        super().__init__()
        self.action_space = Discrete(3)
        self.observation_space = Discrete(11)
        self.nr_indexed = 0
        self.nr_steps = 0
        self.log = []
        print('Init')
    
    def close(self):
        print('Close')

    def reset(self, seed):
        """ Reset database state by removing indexes. 
        
        Returns:
            Observation (nr. indexed batches), information
        """
        self.nr_indexed = 0
        self.nr_steps = 0
        return 0, {}
    
    def step(self, action):
        """ Perform action (add or drop index or no-op).
        
        Args:
            action: add index (1), drop index (2), or no-op.
        
        Returns:
            reward, observation, termination, truncation.
        """
        self.nr_steps += 1
        if action == 1:
            self._add_index()
            print(f'Index one more batch ({self.nr_indexed} indexed)')
        elif action == 2:
            self._drop_index()
            print(f'Index one less batch ({self.nr_indexed} indexed)')
        
        reward = self._reward()
        self.log += [(self.nr_steps, self.nr_indexed)]
        return reward, self.nr_indexed, False, False, {}
    
    def _add_index(self):
        """ Simulates adding one index. """
        self.nr_indexed = min(10, self.nr_indexed+1)
        
    def _drop_index(self):
        """ Simulates dropping one index. """
        self.nr_indexed = max(0, self.nr_indexed-1)
    
    def _reward(self):
        """ Calculates reward value.
        
        Returns:
            Calculated reward value.
        """
        return self.nr_indexed / 10.0