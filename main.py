import os
import pickle
from datetime import datetime

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')
matplotlib.rcParams.update({"axes.grid": False})

from market import *
from allocation import *

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
    'algorithms': ['UCBSmoothed', 'UCBClipped', 'UCBHalf', 'UCBThreeQuarters'],
    'lam': 1
}


def plots(fp, results, constants):
    # Plotting
    for i in range(len(constants['algorithms'])):
        plt.plot(np.arange(constants['max_timesteps']),
                 np.mean(results['rewards'][i], axis=1),
                 label=constants['algorithms'][i])
    plt.title('Rewards over Time')
    plt.xlabel('Timestamp')
    plt.xlabel('Reward')
    plt.legend()
    plt.savefig(f'{fp}/figures/rewards.svg', format='svg', transparent=True, dpi=400)
    plt.savefig(f'{fp}/figures/rewards.png', dpi=400)
    plt.clf()

    for i in range(len(constants['algorithms'])):
        plt.plot(np.arange(constants['max_timesteps']),
                 np.mean(np.cumsum(results['opt_sols'] - results['rewards'][i], axis=0), axis=1),
                 label=constants['algorithms'][i])
    plt.title('Surplus Reward over Time')
    plt.xlabel('Timestamp')
    plt.xlabel('Surplus Reward')
    plt.legend()
    plt.savefig(f'{fp}/figures/surplusRewards.svg', format='svg', transparent=True, dpi=400)
    plt.savefig(f'{fp}/figures/surplusRewards.png', dpi=400)
    plt.clf()

    for i in range(len(constants['algorithms'])):
        plt.plot(np.arange(constants['max_timesteps']),
                 np.mean(results['dissatisfactions'][i], axis=1),
                 label=constants['algorithms'][i])
    plt.title('Dissatisfaction over Time')
    plt.xlabel('Timestamp')
    plt.xlabel('Dissatisfaction')
    plt.legend()
    plt.savefig(f'{fp}/figures/dissatisfactions.svg', format='svg', transparent=True, dpi=400)
    plt.savefig(f'{fp}/figures/dissatisfactions.png', dpi=400)
    plt.clf()

    for i in range(len(constants['algorithms'])):
        plt.plot(np.arange(constants['max_timesteps']),
                 np.mean(np.cumsum(
                     (results['opt_sols'] - results['rewards'][i] + constants['lam'] * results[
                         'dissatisfactions'][i]),
                     axis=0),
                     axis=1),
                 label=constants['algorithms'][i])
    plt.title('Surplus Reward Weighted Dissatisfaction over Time')
    plt.xlabel('Timestamp')
    plt.xlabel('Surplus Reward Weighted Dissatisfaction')
    plt.legend()
    plt.savefig(f'{fp}/figures/surplusRewardsWeightedDissatisfactions.svg', format='svg', transparent=True, dpi=400)
    plt.savefig(f'{fp}/figures/surplusRewardsWeightedDissatisfactions.png', dpi=400)
    plt.clf()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Instantiating experiment...')
    # Creates variables from constants dictionary
    for key, val in constants.items():
        if isinstance(val, str):
            exec(key + f'="{val}"')
        else:
            exec(key + f'={val}')

    # Save Directory
    dt = datetime.now()
    ts = '{}-{}-{} {}:{:02d}'.format(dt.month, dt.day, str(dt.year)[:2], dt.hour, dt.minute)
    fp = f'experiments/{ts}'
    if not os.path.isdir(fp):
        os.makedirs(fp)
    if not os.path.isdir(f'{fp}/figures'):
        os.makedirs(f'{fp}/figures')
    with open(f'{fp}/constants.pkl', 'wb') as handle:
        pickle.dump(constants, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Instantiate metrics
    rewards = np.zeros((len(algorithms), max_timesteps, n_trials))
    dissatisfactions = np.zeros((len(algorithms), max_timesteps, n_trials))
    acceptance_rates = np.zeros((len(algorithms), max_timesteps, n_trials))
    opt_sols = np.zeros(n_trials)

    # Run experiments
    print('Starting trials...')
    for trial in range(n_trials):
        print(f'\tConducting trial {trial + 1}')
        # Generate seeded market
        market = Market(n_users, n_arms,
                        sample_proc=sample_proc,
                        max_util=max_util,
                        seed=seed)

        # Compute optimal solution (if true utilities were known)
        opt_sol = OptSolution(market)
        opt_sol.allocate()
        opt_sols[trial] = opt_sol.surplus()
        market.clear()

        # Run experiments
        for algo_id, algo in enumerate(algorithms):
            print(f'\t\tTesting algorithm {algo}')
            # Instantiate algorithm
            alloc = eval(algo)(market)

            for ts in range(max_timesteps):
                alloc.allocate()
                rewards[algo_id, ts, trial] = alloc.surplus()
                dissatisfactions[algo_id, ts, trial] = alloc.dissatisfaction()
                acceptance_rates[algo_id, ts, trial] = alloc.acceptance_rate()

            market.clear()

        seed = seed + 1

    # Save results
    results = {'rewards': rewards,
               'dissatisfactions': dissatisfactions,
               'acceptance_rates': acceptance_rates,
               'opt_sols': opt_sols}
    with open(f'{fp}/results.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plotting
    print('Plotting...')
    plots(fp, results, constants)
    print('Experiment complete!')
