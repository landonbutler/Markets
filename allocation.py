from abc import ABC, abstractmethod

import numpy as np
from pyomoSolver import *


class Allocation(ABC):
    def __init__(self, market):
        self.market = market

        self.allocation, self.prices, self.arm_counts, self.demand_counts, = None, None, None, None
        self.clear_allocation()

    def return_allocation(self):
        return self.allocation

    def clear_allocation(self):
        self.allocation = -1 * np.ones((self.market.n_users, self.market.max_demand))
        self.validate_allocation()

    @abstractmethod
    def allocate(self, validate=True):
        # Needs to be implemented by all subclasses
        pass

    def surplus(self):
        # TODO implement surplus
        pass

    def dissatisfaction(self):
        # TODO implement dissatisfaction
        pass

    def acceptances(self):
        # TODO implement acceptances
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


# OptSolution: optimal allocation and prices with known utilities
class OptSolution(Allocation):
    def __init__(self, market):
        super().__init__(market)

    def allocate(self, validate=True):
        # Instantiate solver
        solver = PyomoSolver(self.market.n_users, self.market.n_arms, self.market.capacities, self.market.demands)

        # Solve system with true utilities
        self.allocation = solver.solve_system(self.market.utilities)
        self.prices = solver.get_prices() - 1e-8

        if validate:
            self.validate_allocation()