from __future__ import division

from collections import Counter
from itertools import combinations

from future.utils import viewitems, viewkeys  # python2/3 compatibility
from scipy.sparse import csr_matrix


class ConditionalProbability(object):

    def __init__(self):
        self._counts = Counter()
        self._joint_counts = Counter()
        self._n_observations = 0

        self._prob_matrix = None
        self._symb_to_pos = None

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
        if self._n_observations == 0:
            raise ValueError('no observations provided')

        self._symb_to_pos = {symbol: i for i, symbol in enumerate(viewkeys(self._counts))}
        n_symb = len(self._symb_to_pos)

        row_idx = []
        col_idx = []
        prob = []
        for (symbol1, symbol2), joint_count in viewitems(self._joint_counts):
            row_idx.append(self._symb_to_pos[symbol1])
            col_idx.append(self._symb_to_pos[symbol2])
            prob.append(joint_count / self._counts[symbol2])

            row_idx.append(self._symb_to_pos[symbol2])
            col_idx.append(self._symb_to_pos[symbol1])
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
        self._n_observations += 1

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
