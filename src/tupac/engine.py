'''
Created on Mar 5, 2024

@author: immanueltrummer
'''
import psycopg2
import time

from gymnasium import Env
from gymnasium.spaces import Discrete

nr_batches = 10
""" Number of data batches that can be indexed. """


class PgSimEnv(Env):
    """ Environment for Postgres tuning (simulated). """
    
    def __init__(self):
        """ Initializes database for given credentials. """
        super().__init__()
        self.action_space = Discrete(3)
        self.observation_space = Discrete(nr_batches+1)
        self.nr_indexed = 0
        self.nr_steps = 0
        self.log = []
        self.start_s = time.time()
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
        total_s = time.time() - self.start_s
        self.log += [(self.nr_steps, self.nr_indexed, total_s, reward)]
        return reward, self.nr_indexed, False, False, {}
    
    def _add_index(self):
        """ Simulates adding one index. """
        self.nr_indexed = min(nr_batches, self.nr_indexed+1)
        
    def _drop_index(self):
        """ Simulates dropping one index. """
        self.nr_indexed = max(0, self.nr_indexed-1)
    
    def _reward(self):
        """ Calculates reward value.
        
        Returns:
            Calculated reward value.
        """
        return self.nr_indexed / nr_batches


class PgEnv(PgSimEnv):
    """ Tuning environment for Postgres. """
    
    def __init__(self, db, user, password):
        """ Initializes connection to database.
        
        Args:
            db: name of database to tune.
            user: name of database user.
            password: password for database access.
        """
        super().__init__()
        self.connection = psycopg2.connect(
            f'dbname={db} user={user} password={password}')
        self.reset(0)
        self.default_secs = self._benchmark()
    
    def reset(self, seed):
        """ Reset database state by dropping indexes.
        
        Args:
            seed: seed value (not used).
        """
        for i in range(nr_batches):
            sql = f'drop index if exists shipdateindex{i};'
            self._run_sql(sql)
        
        return super().reset(seed)
    
    def close(self):
        """ Close connection to database. """
        self.connection.close()
    
    def _add_index(self):
        """ Index the next data batch. """
        if self.nr_indexed < nr_batches:
            sql = (
                f'create index if not exists shipdateindex{self.nr_indexed} ' 
                f'on lineitem{self.nr_indexed}(l_shipdate);')
            self._run_sql(sql)
            self.nr_indexed += 1
    
    def _benchmark(self):
        """ Benchmark execution time for a simple query.
        
        Returns:
            Execution time for query in seconds.
        """
        sub_queries = []
        for i in range(nr_batches):
            sub_queries += [f'(select max(l_shipdate) from lineitem{i})']
        sql = 'select greatest(' + ' ,'.join(sub_queries) + ');'
        start_s = time.time()
        self._run_sql(sql)
        total_s = time.time() - start_s
        print(f'Benchmark execution time: {total_s}')
        return total_s
    
    def _drop_index(self):
        """ Drop index for one data batch. """
        if self.nr_indexed > 0:
            sql = f'drop index if exists shipdateindex{self.nr_indexed}'
            self._run_sql(sql)
            self.nr_indexed -= 1
    
    def _reward(self):
        """ Calculate reward by benchmarking simple query.
        
        Returns:
            a reward value.
        """
        cur_secs = self._benchmark()
        reward = (self.default_secs - cur_secs) / self.default_secs
        reward = max(0, reward)
        return reward
    
    def _run_sql(self, sql):
        """ Run an SQL query on the database.
        
        Args:
            sql: SQL query to execute.
        """
        print(f'About to execute query: {sql}')
        with self.connection.cursor() as cursor:
            cursor.execute(sql)