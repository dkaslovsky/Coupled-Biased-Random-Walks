from __future__ import division

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from six import iteritems
from six.moves import zip

from coupled_biased_random_walks.count import (ObservationCounter,
                                               get_feature_name, get_mode)
from coupled_biased_random_walks.matrix import (random_walk,
                                                row_normalize_csr_matrix)


class CBRW(object):

    # random walk parameters
    PRESET_RW_PARAMS = {
        'alpha':    0.95,  # damping
        'err_tol':  1e-3,  # convergence criterion for stationary probability
        'max_iter': 100    # max number of steps to take
    }

    def __init__(self, rw_params=None, ignore_unknown=False):
        """
        :param rw_params: random walk parameters to override defaults
        :param ignore_unknown: if True, score an observation containing unknown feature names
        or values based only on features seen during training; if False, score such an observation
        as nan (default)
        """
        self.rw_params = rw_params if rw_params else self.PRESET_RW_PARAMS
        self._unknown_feature_score = 0 if ignore_unknown else np.nan

        self._counter = ObservationCounter()
        self._stationary_prob = None
        self._feature_relevance = None
    
    @property
    def feature_weights(self):
        return self._feature_relevance

    def add_observations(self, observation_iterable):
        """
        Add observations to be modeled
        :param observation_iterable: list of dicts with each dict representing an observation
        taking the form {feature_name: categorical_level/feature_value, ...}
        """
        self._counter.update(observation_iterable)
        return self

    def fit(self):
        """
        Compute model based on current observations in state
        """
        # check number of observations added
        n_observed = get_mode(self._counter.n_obs)
        if n_observed == 0:
            raise ValueError('no observations provided')

        # execute biased random walk
        transition_matrix = self._compute_biased_transition_matrix()
        pi = random_walk(transition_matrix, **self.rw_params).ravel()

        stationary_prob = {}
        feature_relevance = defaultdict(int)

        for feature, idx in iteritems(self._counter.index):
            prob = pi[idx]
            stationary_prob[feature] = prob
            feature_relevance[get_feature_name(feature)] += prob
        # feature relevance scores are to be used as weights; accordingly the paper
        # normalizes them to sum to 1, however this sum normalization should not be
        # necessary since sum(pi) = 1 by definition
        self._stationary_prob = stationary_prob
        self._feature_relevance = dict(feature_relevance)
        return self

    def score(self, observation_iterable):
        """
        Compute an anomaly score for each observation in observation_iterable
        :param observation_iterable: iterable of dict observations with each dict
        taking the form {feature_name: feature_value, ...}
        """
        if not (self._feature_relevance and self._stationary_prob):
            raise ValueError('must call fit method to train on added observations before scoring')
        if isinstance(observation_iterable, dict):
            observation_iterable = [observation_iterable]
        return np.array([self._score(obs) for obs in observation_iterable])

    def _score(self, observation):
        """
        Computes the weighted anomaly score for an observation
        :param observation: dict of the form {feature_name: feature_value, ...}
        """
        return sum(self._get_feature_relevance(item) * \
                   self._stationary_prob.get(item, self._unknown_feature_score)
                    for item in iteritems(observation))

    def _get_feature_relevance(self, feature_tuple):
        """
        Getter for the relevance (weight) of a feature (category)
        :param feature_tuple:  tuple of the form (feature_name, feature_value)
        """
        feature_name = get_feature_name(feature_tuple)
        return self._feature_relevance.get(feature_name, 0)
    
    def _compute_biased_transition_matrix(self):
        """
        Computes biased probability transition matrix of conditional probabilities
        """
        idx = []
        prob = []
        
        bias_dict = self._compute_biases()
        
        for (symbol1, symbol2), joint_count in iteritems(self._counter.joint_counts):

            # get index for symbols
            symb1_idx = self._counter.index[symbol1]
            symb2_idx = self._counter.index[symbol2]

            # get individual counts for symbols
            symb1_count = self._counter.get_count(symbol1)
            symb2_count = self._counter.get_count(symbol2)

            # p(symb1 | symb2)
            p =  bias_dict[symbol2] * joint_count / symb2_count
            if p > 0:
                prob.append(p)
                idx.append((symb1_idx, symb2_idx))

            # p(symb2 | symb1)
            p = bias_dict[symbol1] * joint_count / symb1_count
            if p > 0:
                prob.append(p)
                idx.append((symb2_idx, symb1_idx))

        n_symb = len(self._counter.index)
        trans_matrix = csr_matrix((prob, zip(*idx)), shape=(n_symb, n_symb))
        return row_normalize_csr_matrix(trans_matrix)

    def _compute_biases(self):
        """
        Computes bias for random walk for each feature tuple
        """
        bias_dict = {}
        for feature_name, value_counts in iteritems(self._counter.counts):
            mode = get_mode(value_counts)
            base = 1 - (mode / self._counter.n_obs[feature_name])
            bias_dict.update({feature_val: (1 - (count / mode) + base) / 2
                              for feature_val, count in iteritems(value_counts)})
        return bias_dict
