from __future__ import division

import numpy as np
from scipy.sparse import csr_matrix
from six import iteritems
from six.moves import range, zip

from count import ObservationCounter


class CBRW(object):

    PRESET_RW_PARAMS = {
        'alpha':    0.95,
        'err_tol':  1e-3,
        'max_iter': 100
    }

    def __init__(self, rw_params=None):
        self.rw_params = rw_params if rw_params else self.PRESET_RW_PARAMS
        self._counter = ObservationCounter()
        self._transition_matrix = None
        self._bias_dict = None
        self._stationary_prob = None

    def add_observations(self, observation_iterable):
        self._counter.update(observation_iterable)

    def fit(self):
        if self._counter.n_obs == 0:
            raise ValueError('no observations provided')
        self._bias_dict = self._compute_biases()
        self._transition_matrix = self._compute_transition_matrix()
        self._stationary_prob = self._compute_stationary_prob()
        return self

    def score(self, observation_iterable):
        if isinstance(observation_iterable, dict):
            observation_iterable = [observation_iterable]
        return np.array([self._score(obs) for obs in observation_iterable])

    def _score(self, observation):
        return sum(self._get_node_score(item) for item in iteritems(observation))

    def _get_node_score(self, node_name):
        try:
            return self._stationary_prob[node_name]
        except KeyError:
            raise ValueError('unknown feature value: {}'.format(node_name))

    def _compute_transition_matrix(self):
        idx = []
        prob = []
        for (symbol1, symbol2), joint_count in iteritems(self._counter.joint_counts):

            # get index for symbols
            symb1_idx = self._counter.index[symbol1]
            symb2_idx = self._counter.index[symbol2]

            # get individual counts for symbols
            symb1_count = self._counter.get_count(symbol1)
            symb2_count = self._counter.get_count(symbol2)

            # p(symb1 | symb2)
            idx.append((symb1_idx, symb2_idx))
            prob.append(self._bias_dict[symbol2] * joint_count / symb2_count)
            # p(symb2 | symb1)
            idx.append((symb2_idx, symb1_idx))
            prob.append(self._bias_dict[symbol1] * joint_count / symb1_count)

        n_symb = len(self._counter.index)
        trans_matrix = csr_matrix((prob, zip(*idx)), shape=(n_symb, n_symb))
        return self._row_normalize_csr_matrix(trans_matrix)

    def _compute_biases(self):
        bias_dict = {}
        for feature_name, value_counts in iteritems(self._counter.counts):
            mode = self._get_mode(value_counts)
            base = 1 - mode / self._counter.n_obs
            bias_dict.update({feature_val: self._compute_bias(count, mode, base)
                              for feature_val, count in iteritems(value_counts)})
        return bias_dict

    def _compute_stationary_prob(self):
        stationary_prob = self._random_walk().ravel()
        return {feature_val: stationary_prob[idx]
                for feature_val, idx in iteritems(self._counter.index)}

    def _random_walk(self):
        # get random walk parameters
        alpha = self.rw_params['alpha']
        err_tol = self.rw_params['err_tol']
        max_iter = self.rw_params['max_iter']

        # shape of transition matrix will be length of vectors
        n = self._transition_matrix.shape[0]
        # damping vector
        damping_vec = ((1 - alpha) / n) * np.ones((n, 1))
        # stationary vector initialization
        pi = (1 / n) * np.ones((n, 1))

        for _ in range(max_iter):
            pi_next = damping_vec + alpha * self._transition_matrix.T.dot(pi)
            err = np.linalg.norm(pi - pi_next, ord=np.inf)
            if err <= err_tol:
                return pi_next
            pi = pi_next
        return pi

    @staticmethod
    def _compute_bias(count, mode, base):
        dev = 1 - count / mode
        return 0.5 * (dev + base)

    @staticmethod
    def _get_mode(counter_obj):
        return counter_obj.most_common(1)[0][1]

    @staticmethod
    def _row_normalize_csr_matrix(matrix):
        """
        Row normalize a csr matrix without mutating the input
        :param matrix:
        """
        # get row index for every nonzero element in matrix
        row_idx, col_idx = matrix.nonzero()
        # compute runraveled row sums
        row_sums = matrix.sum(axis=1).A1
        # divide data by (broadcasted) row sums
        normalized = matrix.data / row_sums[row_idx]
        return csr_matrix((normalized, (row_idx, col_idx)), shape=matrix.shape)
