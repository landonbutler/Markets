import numpy as np


class Allocation:
    def __init__(self, market):
        self.market = market

        self.allocation, self.item_counts, self.demand_counts = None, None, None
        self.clear_allocation()

    def clear_market(self):
        self.market = self.market.clear

    def return_allocation(self):
        return self.allocation

    def clear_allocation(self):
        self.allocation = -1 * np.ones((self.market.n_users, self.market.max_demand))
        self.validate_allocation()

    def allocate(self, validate=True):
        # Write own allocation procedure for each allocation algorithm
        if validate:
            self.validate_allocation()

    def count_items(self):
        # Counts the number of times each item is allocated, returns in a vector with
        # item count in the corresponding index
        unique, counts = np.unique(self.allocation, return_counts=True)
        item_counts = np.zeros(self.market.n_items)
        if np.any(unique > -1):
            item_counts[unique] = counts
        self.item_counts = item_counts

    def count_demand(self):
        self.demand_counts = np.count_nonzero(self.allocation + 1, axis=1)

    def validate_allocation(self):
        assert self.allocation.shape[0] == self.market.n_users, 'allocation is not equal in length to n_users'
        assert self.allocation.shape[1] == self.market.max_demand, 'allocation does not support current max_demand'
        assert np.min(self.allocation) >= -1, 'allocation contains negative values < -1'
        assert np.max(self.allocation) <= self.market.n_items, 'allocation contains allocated values > n_items'

        # Check that no item is allocated more than its capacity
        self.count_items()
        assert np.all(self.item_counts <= self.market.capacities), 'an item is allocated more than its capacity'

        # Check that no user is allocated more items than their demand
        self.count_demand()
        assert np.all(self.demand_counts <= self.market.demands), 'a user is allocated more items than its demand'


# Greedy_UCB: all users get allocated item with highest UCB, regardless of overlaps
class GreedyUCB(Allocation):
    def __init__(self, market):
        super().__init__(market)

    def allocate(self, validate=True):
        # TODO: Make sure this overrides properly
        print('here')
        # Get top item for each user
        allocation = np.argmax(self.market.upp_conf, axis=1)
        print(allocation)
        # Unallocate everyone (necessary for multi-item allocation)
        self.clear_allocation()

        # Fill in allocation
        self.allocation[:, 0] = allocation

        # Will likely fail due to clashes
        super.allocate(validate)
