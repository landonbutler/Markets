from market import *
from learningEnv import *
from stable_baselines3.common.env_checker import check_env

# Experiment constants
constants = {
    'max_timesteps': 200,
    'n_trials': 10,
    'n_users': 50,
    'n_arms': 40,
    'noise_param': 1,
    'max_util': 1,
    'sample_proc': "beta",
    'seed': 34,
    'models': ['ClippedWalrasian'],
    'lam': 1
}

if __name__ == '__main__':
    print('Instantiating experiment...')
    # Creates variables from constants dictionary
    for key, val in constants.items():
        if isinstance(val, str):
            exec(key + f'="{val}"')
        else:
            exec(key + f'={val}')

    for model in models:
        # Generate seeded market
        market = Market(n_users, n_arms,
                        sample_proc=sample_proc,
                        max_util=max_util,
                        seed=seed)

        env = eval(model + 'Env')(market)
        check_env(env, warn=True)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # print(obs)
