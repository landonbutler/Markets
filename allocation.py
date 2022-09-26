from abc import ABC, abstractmethod

import numpy as np


class Allocation(ABC):
    def __init__(self, market):
        self.market = market

        self.allocation, self.arm_counts, self.demand_counts = None, None, None
        self.clear_allocation()

    def clear_market(self):
        self.market = self.market.clear

    def return_allocation(self):
        return self.allocation

    def clear_allocation(self):
        self.allocation = -1 * np.ones((self.market.n_users, self.market.max_demand))
        self.validate_allocation()

    @abstractmethod
    def allocate(self, validate=True):
        # Needs to be implemented by all subclasses
        pass

    def count_arms(self):
        # Counts the number of times each arm is allocated, returns in a vector with
        # arm count in the corresponding index
        unique, counts = np.unique(self.allocation, return_counts=True)
        arm_counts = np.zeros(self.market.n_arms)
        if np.any(unique > -1):
            arm_counts[unique.astype(int)] = counts
        self.arm_counts = arm_counts

    def count_demand(self):
        self.demand_counts = np.count_nonzero(self.allocation + 1, axis=1)

    def validate_allocation(self):
        assert self.allocation.shape[0] == self.market.n_users, 'allocation is not equal in length to n_users'
        assert self.allocation.shape[1] == self.market.max_demand, 'allocation does not support current max_demand'
        assert np.min(self.allocation) >= -1, 'allocation contains negative values < -1'
        assert np.max(self.allocation) <= self.market.n_arms, 'allocation contains allocated values > n_arms'

        # Check that no arm is allocated more than its capacity
        self.count_arms()
        assert np.all(self.arm_counts <= self.market.capacities), 'an arm is allocated more than its capacity'

        # Check that no user is allocated more arms than their demand
        self.count_demand()
        assert np.all(self.demand_counts <= self.market.demands), 'a user is allocated more arms than its demand'


# Greedy_UCB: all users get allocated arm with highest UCB, regardless of overlaps
class GaleShapley(Allocation):
    def __init__(self, market):
        super().__init__(market)

    def allocate(self, validate=True):
        # Unallocate everyone (necessary for multi-arm allocation)
        self.clear_allocation()

        cur_utilities = self.market.utilities.copy()
        cur_arm_match = -1 * np.ones(self.market.n_arms, dtype=int)

        # Only 1 arm allocated to everyone, so just check 1st column
        # If any user is unallocated, continue matching
        while np.any(self.allocation[:, 0] == -1):
            print('loop')
            # Get top arm for each unmatched user based on their current utilities
            unmatched_users = np.arange(self.market.n_users)[self.allocation[:, 0] == -1]
            print(unmatched_users)
            top_arm = np.argmax(cur_utilities[unmatched_users], axis=1)
            for i, arm_id in enumerate(top_arm):
                user_id = unmatched_users[i]
                # If proposed arm prefers new user, will match with new user
                old_match = cur_arm_match[arm_id]
                print(user_id, arm_id, old_match)
                if old_match == -1:

                    # Arm is unmmatched, so match it with new user
                    self.allocation[user_id, 0] = arm_id
                    cur_arm_match[arm_id] = user_id
                    print(f'Matched user {user_id} with arm {arm_id}')
                elif self.market.arm_prefs[user_id, arm_id] > self.market.arm_prefs[old_match, arm_id]:
                    # Arm prefers new user over the old user
                    self.allocation[user_id, 0] = arm_id
                    cur_arm_match[arm_id] = user_id
                    self.allocation[old_match, 0] = -1
                    cur_utilities[old_match, arm_id] = -1
                    print(f'Matched user {user_id} with arm {arm_id}, unmatched {old_match}')
                else:
                    # Arm prefers old user, so new user will never get matched to it
                    cur_utilities[user_id, arm_id] = -1
                    print(f'Failed matching {user_id} with arm {arm_id}')
        if validate:
            self.validate_allocation()


# Greedy_UCB: all users get allocated arm with highest UCB, regardless of overlaps
class GreedyUCB(Allocation):
    def __init__(self, market):
        super().__init__(market)

    def allocate(self, validate=True):
        # Get top arm for each user
        allocation = np.argmax(self.market.upp_conf, axis=1)

        # Unallocate everyone (necessary for multi-arm allocation)
        self.clear_allocation()

        # Fill in allocation
        self.allocation[:, 0] = allocation

        if validate:
            self.validate_allocation()
