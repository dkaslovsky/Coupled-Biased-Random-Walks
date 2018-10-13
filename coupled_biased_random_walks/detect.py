from __future__ import division

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from six import iteritems
from six.moves import zip

from coupled_biased_random_walks.count import (ObservationCounter,
                                               get_feature_name, get_mode)
from coupled_biased_random_walks.matrix import (dict_to_csr_matrix,
                                                random_walk,
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
            raise CBRWFitError('must add observations before calling fit method')

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
            raise CBRWScoreError()
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
        prob_idx = {}
        
        bias_dict = self._compute_biases()
        
        for (feature1, feature2), joint_count in iteritems(self._counter.joint_counts):

            # get index for features
            feature1_idx = self._counter.index[feature1]
            feature2_idx = self._counter.index[feature2]

            # get individual counts for features
            feature1_count = self._counter.get_count(feature1)
            feature2_count = self._counter.get_count(feature2)

            # p(feature1 | feature2)
            p = bias_dict[feature2] * joint_count / feature2_count
            if p > 0:
                prob_idx[(feature1_idx, feature2_idx)] = p

            # p(feature2 | feature1)
            p = bias_dict[feature1] * joint_count / feature1_count
            if p > 0:
                prob_idx[(feature2_idx, feature1_idx)] = p

        # raise exception on empty probability-index dict
        if not prob_idx:
            raise CBRWFitError('all biased joint probabilities are zero')
        
        # construct sparse matrix
        n_features = len(self._counter.index)
        trans_matrix = dict_to_csr_matrix(prob_idx, shape=n_features)
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


class CBRWFitError(Exception):
    pass


class CBRWScoreError(Exception):

    message = 'must call fit method to train on added observations before scoring'
    
    def __str__(self):
        return self.message
