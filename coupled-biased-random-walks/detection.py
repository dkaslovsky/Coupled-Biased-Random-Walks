from __future__ import division

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from six import iteritems
from six.moves import range, zip

from count import ObservationCounter, get_feature_name, get_mode


class CBRW(object):

    PRESET_RW_PARAMS = {
        'alpha':    0.95,
        'err_tol':  1e-3,
        'max_iter': 100
    }

    def __init__(self, rw_params=None):
        self.rw_params = rw_params if rw_params else self.PRESET_RW_PARAMS
        self._counter = ObservationCounter()
        self._stationary_prob = None
        self._feature_relevance = None

    def add_observations(self, observation_iterable):
        self._counter.update(observation_iterable)

    def fit(self):
        if self._counter.n_obs == 0:
            raise ValueError('no observations provided')

        # execute biased random walk
        bias_dict = self._compute_biases()
        transition_matrix = self._compute_biased_transition_matrix(bias_dict)
        pi = self._random_walk(transition_matrix, self.rw_params).ravel()

        stationary_prob = {}
        feature_relevance = defaultdict(int)

        for feature_val, idx in iteritems(self._counter.index):
            prob = pi[idx]
            stationary_prob[feature_val] = prob
            feature_relevance[get_feature_name(feature_val)] += prob
        # feature relevance scores are to be used as weights; accordingly the paper
        # normalizes them to sum to 1, however this sum normalization should not be
        # necessary since sum(pi) = 1 by definition
        self._stationary_prob, self._feature_relevance = stationary_prob, dict(feature_relevance)
        return self

    def score(self, observation_iterable):
        if isinstance(observation_iterable, dict):
            observation_iterable = [observation_iterable]
        return np.array([self._score(obs) for obs in observation_iterable])

    def _score(self, observation):
        return sum(self._get_feature_relevance(item) * self._get_node_score(item)
                   for item in iteritems(observation))

    def _get_node_score(self, node_name):
        try:
            return self._stationary_prob[node_name]
        except KeyError:
            raise ValueError('unknown feature value: {}'.format(node_name))

    def _get_feature_relevance(self, feature_tuple):
        feature_name = get_feature_name(feature_tuple)
        return self._feature_relevance.get(feature_name, 0)

    def _compute_biased_transition_matrix(self, bias_dict):
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
            prob.append(bias_dict[symbol2] * joint_count / symb2_count)
            # p(symb2 | symb1)
            idx.append((symb2_idx, symb1_idx))
            prob.append(bias_dict[symbol1] * joint_count / symb1_count)

        n_symb = len(self._counter.index)
        trans_matrix = csr_matrix((prob, zip(*idx)), shape=(n_symb, n_symb))
        return self._row_normalize_csr_matrix(trans_matrix)

    def _compute_biases(self):
        bias_dict = {}
        for feature_name, value_counts in iteritems(self._counter.counts):
            mode = get_mode(value_counts)
            base = 1 - mode / self._counter.n_obs
            bias_dict.update({feature_val: self._compute_bias(count, mode, base)
                              for feature_val, count in iteritems(value_counts)})
        return bias_dict

    @staticmethod
    def _random_walk(transition_matrix, params):
        try:
            # get random walk parameters
            alpha = params['alpha']
            err_tol = params['err_tol']
            max_iter = params['max_iter']
        except KeyError:
            raise ValueError('one or more of alpha, err_tol, and max_iter '
                             'missing from ranom walk parameter dict')

        # shape of transition matrix will be length of vectors
        n = transition_matrix.shape[0]
        # damping vector
        damping_vec = ((1 - alpha) / n) * np.ones((n, 1))
        # stationary vector initialization
        pi = (1 / n) * np.ones((n, 1))

        for _ in range(max_iter):
            pi_next = damping_vec + alpha * transition_matrix.T.dot(pi)
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
    def _row_normalize_csr_matrix(matrix):
        """
        Row normalize a csr matrix without mutating the input
        :param matrix: scipy.sparse.csr_matrix instance
        """
        # get row index for every nonzero element in matrix
        row_idx, col_idx = matrix.nonzero()
        # compute runraveled row sums
        row_sums = matrix.sum(axis=1).A1
        # divide data by (broadcasted) row sums
        normalized = matrix.data / row_sums[row_idx]
        return csr_matrix((normalized, (row_idx, col_idx)), shape=matrix.shape)
