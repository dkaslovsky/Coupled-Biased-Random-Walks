from __future__ import division

from collections import Counter
from itertools import combinations

from future.utils import viewitems  # instead of iteritems()/items() for python2/3 compatibility
from scipy.sparse import csr_matrix


class FixedValueDict(dict):

    def __setitem__(self, key, value):
        if dict.has_key(self, key):
            raise KeyError('cannot overwrite key {}'.format(key))
        dict.__setitem__(self, key, value)


class IncrementingValueDict(object):

    def __init__(self):
        self._next_val = 0
        self._d = FixedValueDict()

    def insert(self, key):
        try:
            self._d[key] = self._next_val
        except KeyError:
            return
        self._next_val += 1

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getitem__(self, key):
        return self._d.__getitem__(key)

    def __len__(self):
        return self._d.__len__()

    def __repr__(self):
        return self._d.__repr__()


class ConditionalProbability(object):

    def __init__(self):
        self._counts = Counter()
        self._joint_counts = Counter()
        self._symb_to_pos = IncrementingValueDict()
        self._prob_matrix = None

    @property
    def prob_matrix(self):
        return self._prob_matrix

    @property
    def symb_to_pos(self):
        return self._symb_to_pos

    def observe(self, observation_iterable):
        for observation in observation_iterable:
            if not isinstance(observation, (list, tuple)):
                raise ValueError('observation must be of type list or tuple')
            self._count_observation(observation)

    def compute_prob(self):
        n_symb = len(self._symb_to_pos)
        if n_symb == 0:
            raise ValueError('no observations provided')

        prob = []
        row_idx = []
        col_idx = []
        for (symbol1, symbol2), joint_count in viewitems(self._joint_counts):

            symb1_idx = self._symb_to_pos[symbol1]
            symb2_idx = self._symb_to_pos[symbol2]

            row_idx.append(symb1_idx)
            col_idx.append(symb2_idx)
            prob.append(joint_count / self._counts[symbol2])

            row_idx.append(symb2_idx)
            col_idx.append(symb1_idx)
            prob.append(joint_count / self._counts[symbol1])

        self._prob_matrix = csr_matrix((prob, (row_idx, col_idx)), shape=(n_symb, n_symb))

    def get_prob(self, u, v):
        if self.prob_matrix is not None and self._symb_to_pos is not None:
            row = self._symb_to_pos.get(u, None)
            col = self._symb_to_pos.get(v, None)
            if row is not None and col is not None:
                return self.prob_matrix[row, col]
            raise ValueError('unknown symbol')
        raise ValueError('must call compute_prob_matrix first')

    def _count_observation(self, observation):
        obs = set(observation)
        self._update_counts(obs)
        self._update_joint_counts(obs)
        self._update_symb_to_pos(obs)

    def _update_counts(self, obs):
        """

        :param obs: iterable of *unique* values
        """
        self._counts.update(obs)

    def _update_joint_counts(self, obs):
        """

        :param obs: iterable of *unique* values
        """
        pairs = combinations(sorted(obs), 2)
        self._joint_counts.update(pairs)

    def _update_symb_to_pos(self, observation):
        for symbol in observation:
            self._symb_to_pos.insert(symbol)
