import numpy as np


class Market:
    def __init__(self, n_users, n_items, demands=None, capacities=None, sample_proc=None, max_util=1):
        assert 0 < n_users == int(n_users), 'n_users not a positive integer'
        assert 0 < n_items == int(n_items), 'n_items not a positive integer'

        # If capacities or demands are not supplied, set equal to 1
        if not demands:
            demands = np.ones(n_users)
        if not capacities:
            capacities = np.ones(n_items)
        if not sample_proc:
            print('\tUsing Beta distribution for sampling utilities')
            sample_proc = 'beta'

        # Assert that quantities are correct
        assert len(demands) == n_users, 'demands are not the same length as n_users'
        assert np.all(demands >= 0), 'at least one user demand is negative'
        assert len(capacities) == n_items, 'capacities are not the same length as n_items'
        assert np.all(capacities >= 0), 'some item capacities are negative'
        assert sample_proc in {'beta', 'uniform'}, 'sample_proc not in available distributions'
        assert max_util > 0, 'max_util not positive'

        self.n_users = n_users
        self.n_items = n_items
        self.demands = demands
        self.max_demand = int(np.max(self.demands))
        self.capacities = capacities
        self.max_util = max_util
        self.utilities = None
        self.sample_proc = sample_proc

        self.sample_utilities()

        self.low_conf = np.zeros((self.n_users, self.n_items))
        self.upp_conf = self.max_util * np.ones((self.n_users, self.n_items))

    def sample_utilities(self):
        if self.sample_proc == 'beta':
            self.utilities = np.random.beta(6, 6, size=(self.n_users, self.n_items))
        elif self.sample_proc == 'uniform':
            self.utilities = np.random.uniform(0, self.max_util, size=(self.n_users, self.n_items))

    def clear(self):
        # Resample utilities and restart confidence intervals
        self.sample_utilities()

        self.low_conf = np.zeros((self.n_users, self.n_items))
        self.upp_conf = self.max_util * np.ones((self.n_users, self.n_items))
