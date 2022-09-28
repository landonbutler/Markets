from abc import ABC, abstractmethod

import numpy as np

np.seterr(over='ignore')
from pyomoSolver import *


class Allocation(ABC):
    def __init__(self, market):
        self.market = market

        self.allocation, self.acc_alloc = None, None
        self.prices = None
        self.arm_counts, self.demand_counts = None, None

        # Instantiate an empty allocation
        self.clear_allocation()

        # Instantiate solver
        self.solver = PyomoSolver(self.market.n_users, self.market.n_arms, self.market.capacities, self.market.demands)

    def return_allocation(self):
        return self.allocation

    def clear_allocation(self):
        self.allocation = np.zeros((self.market.n_users, self.market.n_arms))
        self.validate_allocation()

    @abstractmethod
    def allocate(self, validate=True):
        # Needs to be implemented by all subclasses
        pass

    def surplus(self):
        # Sum of element-wise multiplication of allocation by true utilities
        return np.sum(np.multiply(self.allocation, self.market.utilities))

    def dissatisfaction(self):
        # First, finds current surplus, which is the sum of element-wise
        # multiplication of allocation by the different between the true utilities and the prices
        # Next, finds best surplus, which is the max difference between true utilities and the prices (or 0)
        # Finally, the dissatisfcation is the difference between these two, summed across all users

        current_surplus = np.sum(np.multiply(self.allocation, self.market.utilities - self.prices), axis=1)
        best_surplus = np.maximum(np.max(self.market.utilities - self.prices, axis=1), 0)
        return np.sum(best_surplus - current_surplus)

    def acceptance_rate(self):
        # Acceptance rate of offers
        return np.sum(self.acc_alloc) / np.sum(self.allocation)

    def filter_decisions(self):
        # Determines which arms would be accepted for the given price,
        # then intersects these with the current allocation
        accepted = np.int32(self.market.utilities >= self.prices)
        self.acc_alloc = np.multiply(self.allocation, accepted)

    def get_proxies(self, alpha):
        # Returns a point in the confidence interval to be used as a heuristic for true utility
        return self.market.low_conf + alpha * (self.market.upp_conf - self.market.low_conf)

    def update_conf_intervals(self):
        rej_alloc = self.allocation - self.acc_alloc
        # Updates confidence intervals using allocations that were rejected and accepted
        prices_extended = np.tile(self.prices, (self.market.n_users, 1))
        self.market.low_conf = np.maximum(self.market.low_conf,
                                          self.acc_alloc * prices_extended + (
                                                  1 - self.acc_alloc) * self.market.low_conf)
        self.market.upp_conf = np.minimum(self.market.upp_conf,
                                          rej_alloc * prices_extended + (1 - rej_alloc) * self.market.upp_conf)

    def count_arms(self):
        # Counts the number of users each arm is allotted to
        self.arm_counts = np.sum(self.allocation, axis=0)

    def count_demand(self):
        # Counts the number of arms each user is pulling
        self.demand_counts = np.sum(self.allocation, axis=1)

    def validate_allocation(self):
        assert self.allocation.shape[0] == self.market.n_users, 'allocation is not equal in length to n_users'
        assert self.allocation.shape[1] == self.market.n_arms, 'allocation is not equal in width to n_arms'
        assert np.min(self.allocation) >= 0, 'allocation contains negative values'

        # Check that no arm is allocated more than its capacity
        self.count_arms()
        assert np.all(self.arm_counts <= self.market.capacities), 'an arm is allocated more than its capacity'

        # Check that no user is allocated more arms than their demand
        self.count_demand()
        assert np.all(self.demand_counts <= self.market.demands), 'a user is allocated more arms than its demand'


# OptSolution: optimal allocation and prices with known utilities
class OptSolution(Allocation):
    def __init__(self, market):
        super().__init__(market)

    def allocate(self, validate=True):
        # Solve system with true utilities
        self.allocation = self.solver.solve_system(self.market.utilities)
        self.prices = self.solver.get_prices() - 1e-8
        self.filter_decisions()

        if validate:
            self.validate_allocation()


# UCBHalf: TODO fill in description
class UCBHalf(Allocation):
    def __init__(self, market):
        super().__init__(market)

    def compute_adjusted_prices(self, unadjusted_prices, delta=1e-7):
        p_learn = np.max(np.multiply(self.get_proxies(0.5), self.allocation), axis=0)
        to_learn = np.max(np.multiply(self.allocation, (self.market.upp_conf - self.market.low_conf > delta)),
                          axis=0) > 0
        self.prices = to_learn * p_learn + (1 - to_learn) * unadjusted_prices

    def allocate(self, validate=True):
        utility_proxies = self.get_proxies(1)
        self.allocation = self.solver.solve_system(utility_proxies)
        unadjusted_prices = self.solver.get_prices() - 1e-8

        self.compute_adjusted_prices(unadjusted_prices)

        self.filter_decisions()
        self.update_conf_intervals()

        if validate:
            self.validate_allocation()


# UCBThreeQuarters: TODO fill in description
class UCBThreeQuarters(Allocation):
    def __init__(self, market):
        super().__init__(market)

    def compute_adjusted_prices(self, unadjusted_prices, delta=1e-7):
        p_learn = np.max(np.multiply(self.get_proxies(0.75), self.allocation), axis=0)
        to_learn = np.max(np.multiply(self.allocation, (self.market.upp_conf - self.market.low_conf > delta)),
                          axis=0) > 0
        self.prices = to_learn * p_learn + (1 - to_learn) * unadjusted_prices

    def allocate(self, validate=True):
        utility_proxies = self.get_proxies(1)
        self.allocation = self.solver.solve_system(utility_proxies)
        unadjusted_prices = self.solver.get_prices() - 1e-8

        self.compute_adjusted_prices(unadjusted_prices)

        self.filter_decisions()
        self.update_conf_intervals()

        if validate:
            self.validate_allocation()


# UCBClipped: TODO fill in description
class UCBClipped(Allocation):
    def __init__(self, market):
        super().__init__(market)

    def compute_clipped_prices(self, unclipped_prices):
        extended_prices = np.tile(unclipped_prices, (self.market.n_users, 1))
        A = self.market.low_conf - extended_prices
        A_max = np.max(A, axis=1)
        B = self.market.upp_conf - extended_prices

        other_and_valid = np.multiply(1 - self.allocation, (B.T > A_max).T)
        # B_other_min = np.min(B * other_and_valid + 10 * self.market.max_util * (1 - other_and_valid), axis=1)

        # B_second_max = np.max(B * (1 - self.allocation) - 10 * self.market.max_util * self.allocation, axis=1)

        other_best_surplus = np.max((self.market.upp_conf - unclipped_prices) * (1 - self.allocation), axis=1)
        current_worst_surplus = np.sum((self.market.low_conf - unclipped_prices) * self.allocation, axis=1)

        # current_best_surplus = np.sum((self.market.upp_conf - unsmoothed_prices) * self.allocation, axis=1)

        current_r_lower = np.sum(np.multiply(self.market.low_conf, self.allocation), axis=1)
        current_r_upper = np.sum(np.multiply(self.market.upp_conf, self.allocation), axis=1)

        opt_found = np.bitwise_and(current_worst_surplus - other_best_surplus + 1e-8 > 0,
                                   current_worst_surplus + 1e-8 > 0)

        p_w_user = self.allocation @ unclipped_prices
        p_user = np.zeros(self.market.n_users)
        for n in range(self.market.n_users):
            if opt_found[n]:
                p_user[n] = p_w_user[n] - 1e-8
            else:
                smallest_price = current_r_lower[n] + 0.25 * (current_r_upper[n] - current_r_lower[n])
                largest_price = current_r_lower[n] + 0.75 * (current_r_upper[n] - current_r_lower[n])
                p_user[n] = p_w_user[n]
                p_user[n] = np.minimum(np.maximum(p_user[n], smallest_price), largest_price)

        self.prices = self.allocation.T @ p_user

    def allocate(self, validate=True):
        utility_proxies = self.get_proxies(1)
        self.allocation = self.solver.solve_system(utility_proxies)
        unclipped_prices = self.solver.get_prices() - 1e-8

        self.compute_clipped_prices(unclipped_prices)

        self.filter_decisions()
        self.update_conf_intervals()

        if validate:
            self.validate_allocation()


# UCBSmoothed: TODO fill in description
class UCBSmoothed(Allocation):
    def __init__(self, market):
        super().__init__(market)

    def compute_smoothed_prices(self, unsmoothed_prices):
        extended_prices = np.tile(unsmoothed_prices, (self.market.n_users, 1))
        A = self.market.low_conf - extended_prices
        A_max = np.max(A, axis=1)
        B = self.market.upp_conf - extended_prices

        other_and_valid = np.multiply(1 - self.allocation, (B.T > A_max).T)
        # B_other_min = np.min(B * other_and_valid + 10 * self.market.max_util * (1 - other_and_valid), axis=1)

        # B_second_max = np.max(B * (1 - self.allocation) - 10 * self.market.max_util * self.allocation, axis=1)

        other_best_surplus = np.max((self.market.upp_conf - unsmoothed_prices) * (1 - self.allocation), axis=1)
        current_worst_surplus = np.sum((self.market.low_conf - unsmoothed_prices) * self.allocation, axis=1)

        # current_best_surplus = np.sum((self.market.upp_conf - unsmoothed_prices) * self.allocation, axis=1)

        current_r_lower = np.sum(np.multiply(self.market.low_conf, self.allocation), axis=1)
        current_r_upper = np.sum(np.multiply(self.market.upp_conf, self.allocation), axis=1)

        opt_found = np.bitwise_and(current_worst_surplus - other_best_surplus + 1e-8 > 0,
                                   current_worst_surplus + 1e-8 > 0)

        p_w_user = self.allocation @ unsmoothed_prices
        p_user = np.zeros(self.market.n_users)
        for n in range(self.market.n_users):
            if opt_found[n]:
                p_user[n] = p_w_user[n] - 1e-8
            else:
                p_min = current_r_lower[n] + 0.25 * (current_r_upper[n] - current_r_lower[n])
                p_max = current_r_lower[n] + 0.75 * (current_r_upper[n] - current_r_lower[n])
                p_center = current_r_lower[n] + 0.5 * (current_r_upper[n] - current_r_lower[n])
                p_len = current_r_upper[n] - current_r_lower[n]
                alpha = 4
                p_user[n] = p_w_user[n]
                if p_len > 1e-10:
                    p_user[n] = p_min + (p_max - p_min) / (1 + np.exp(-alpha * (p_user[n] - p_center) / p_len))
                else:
                    p_user[n] = p_min

        self.prices = self.allocation.T @ p_user

    def allocate(self, validate=True):
        utility_proxies = self.get_proxies(1)
        self.allocation = self.solver.solve_system(utility_proxies)
        unsmoothed_prices = self.solver.get_prices() - 1e-8

        self.compute_smoothed_prices(unsmoothed_prices)

        self.filter_decisions()
        self.update_conf_intervals()

        if validate:
            self.validate_allocation()
