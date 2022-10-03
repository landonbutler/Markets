import gym
from gym import spaces
from gym.spaces import Dict
import numpy as np
from allocation import *


def flatten_dict(unflattened_dict):
    flattened_dicts = []
    for key in ['low_conf', 'upp_conf', 'allocation', 'init_prices']:
        print(key)
        print(unflattened_dict[key])
        flattened_dicts.append(unflattened_dict[key].flatten())
    return np.concatenate(flattened_dicts, dtype=np.float64)


class ClippedWalrasianEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, market, lam=1):
        super(ClippedWalrasianEnv, self).__init__()
        self.market = market
        self.allocation = UCBClipped(self.market)
        self.lam = lam
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Tuple((
            spaces.Box(low=0, high=0.5, shape=(1, 1), dtype=np.float64),
            spaces.Box(low=0.5, high=1, shape=(1, 1), dtype=np.float64)))

        # Get current intervals
        # - lower interval
        # - upper interval
        # - allocation
        # - unclipped prices
        n_user_arm_pairs = self.market.n_users * self.market.n_arms
        obs_length = 3 * n_user_arm_pairs + self.market.n_arms
        low = np.zeros(obs_length)
        high = self.market.max_util * np.ones(obs_length)
        high[2 * n_user_arm_pairs:3 * n_user_arm_pairs] = self.market.max_util
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs_length,), dtype=np.float64)

    def step(self, action):
        alpha, beta = action[0], action[1]
        self.allocation.allocate(alpha=alpha, beta=beta)
        observation = {'low_conf': self.market.low_conf, 'upp_conf': self.market.upp_conf,
                       'allocation': self.allocation.allocation, 'init_prices': self.allocation.init_prices}
        reward = self.allocation.surplus() + self.lam * - self.allocation.dissatisfaction()
        done = True
        info = {}
        return flatten_dict(observation), reward, done, info

    def reset(self):
        self.market.clear()
        self.allocation = UCBClipped(self.market)
        observation = {'low_conf': self.market.low_conf, 'upp_conf': self.market.upp_conf,
                       'allocation': self.allocation.allocation, 'init_prices': self.allocation.init_prices}
        return flatten_dict(observation)

    def render(self, mode="human"):
        pass

    def close(self):
        pass
