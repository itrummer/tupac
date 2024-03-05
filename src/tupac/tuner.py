'''
Created on Mar 5, 2024

@author: immanueltrummer
'''
import argparse
import stable_baselines3
import tupac.engine


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('db', type=str, help='Name of database')
    parser.add_argument('user', type=str, help='Database user name')
    parser.add_argument('password', type=str, help='Database password')
    args = parser.parse_args()
    
    env = tupac.engine.PgSimEnv()
    model = stable_baselines3.A2C(
        'MlpPolicy', env, verbose=True, 
        normalize_advantage=True)
    model.learn(total_timesteps=100)
    
    for step, nr_indexed in env.log:
        print(f'{step}\t{nr_indexed}')
    
    # print('Testing Now!!!')
    # env = model.get_env()
    # obs = env.reset()
    # for i in range(10):
        # action, _state = model.predict(obs, deterministic=True)
        # obs, reward, done, info = env.step(action)