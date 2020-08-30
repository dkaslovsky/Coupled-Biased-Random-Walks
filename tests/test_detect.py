import unittest
from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix
from six import itervalues

from coupled_biased_random_walks.count import isnan
from coupled_biased_random_walks.detect import (CBRW, CBRWFitError,
                                                CBRWScoreError)


class TestCBRW(unittest.TestCase):
    """
    Unit tests for CBRW
    """

    observations = [
        {'feature_a': 'a_val_1', 'feature_b': 'b_val_1', 'feature_c': 'c_val_1'},
        {'feature_b': 'b_val_1', 'feature_c': 'c_val_2', 'feature_a': 'a_val_1'}
    ]

    def setUp(self):
        self.cbrw = CBRW()
        self.cbrw.add_observations(self.observations)

    def test_get_feature_relevance(self):
        self.cbrw._feature_relevance = {
            'feature_a': 0.5
        }
        # get relevance for valid feature tuple
        rel = self.cbrw._get_feature_relevance(('feature_a', 'a_val_1'))
        self.assertEqual(rel, 0.5)
        # get relevance for tuple with valid feature name
        rel = self.cbrw._get_feature_relevance(('feature_a', 'xxx'))
        self.assertEqual(rel, 0.5)
        # get relevance for invalid feature name
        rel = self.cbrw._get_feature_relevance(('xxx', 'xxx'))
        self.assertEqual(rel, 0)

    def test_compute_biases(self):
        bias_dict = self.cbrw._compute_biases()
        self.assertEqual(bias_dict[('feature_a', 'a_val_1')], 0)
        self.assertEqual(bias_dict[('feature_b', 'b_val_1')], 0)
        self.assertEqual(bias_dict[('feature_c', 'c_val_1')], 0.25)
        self.assertEqual(bias_dict[('feature_c', 'c_val_2')], 0.25)

    def test_compute_biased_transition_matrix(self):
        transition_matrix = self.cbrw._compute_biased_transition_matrix()
        self.assertIsInstance(transition_matrix, csr_matrix)
        self.assertTupleEqual(transition_matrix.shape, (4, 4))
        self.assertTrue((transition_matrix.data > 0).all())
        self.assertTrue((transition_matrix.data <= 1).all())

    def test_fit_no_data(self):
        self.cbrw = CBRW()
        with self.assertRaises(CBRWFitError):
            self.cbrw.fit()

    def test_fit(self):
        self.cbrw.fit()
        self.assertIsNotNone(self.cbrw._stationary_prob)
        self.assertIsNotNone(self.cbrw._feature_relevance)

    def test_score_before_fit(self):
        with self.assertRaises(CBRWScoreError):
            _ = self.cbrw.score(self.observations)

    def test_score(self):
        self.cbrw.fit()

        # score observation where all features and values
        # have been previously observed
        to_be_scored = self.observations[0]
        score = self.cbrw.score(to_be_scored)
        score = score[0]
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        # actual score is approximately 0.2759 so test this
        # value in case implementation changes
        self.assertAlmostEqual(score, 0.2759, places=4)

    def test_score_unknown_features_default(self):
        self.cbrw.fit()

        # score observation where all features but not all
        # values have been previously observed
        to_be_scored = {
            'feature_a': 'a_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        score = self.cbrw.score(to_be_scored)
        self.assertTrue(isnan(score[0]))

        # score observation where a feature has not
        # been previously observed
        to_be_scored = {
            'feature_x': 'x_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        score = self.cbrw.score(to_be_scored)
        self.assertTrue(isnan(score[0]))

        # score valid and invalid observations in one call
        to_be_scored = [
            self.observations[0],
            {'feature_x': 'x_val_x', 'feature_b': 'b_val_1'}
        ]
        scores = self.cbrw.score(to_be_scored)
        valid_score = scores[0]
        invalid_score = scores[1]
        self.assertFalse(isnan(valid_score))
        self.assertGreaterEqual(valid_score, 0)
        self.assertLessEqual(valid_score, 1)
        self.assertTrue(isnan(invalid_score))

    def test_score_unknown_features_ignore(self):
        self.cbrw = CBRW(ignore_unknown=True)
        self.cbrw.add_observations(self.observations)
        self.cbrw.fit()

        # score observation where all features but not all
        # values have been previously observed
        to_be_scored = {
            'feature_a': 'a_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        actually_scored = {
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        score = self.cbrw.score(to_be_scored)
        actual_score = self.cbrw.score(actually_scored)
        self.assertFalse(isnan(score[0]))
        self.assertEqual(score, actual_score)

        # score observation where a feature has not
        # been previously observed
        to_be_scored = {
            'feature_x': 'x_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        actually_scored = {
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        score = self.cbrw.score(to_be_scored)
        actual_score = self.cbrw.score(actually_scored)
        self.assertFalse(isnan(score[0]))
        self.assertEqual(score, actual_score)

        # score observation where no features have
        # previously been observed
        to_be_scored = {
            'feature_x': 'x_val_x',
            'feature_y': 'y_val_1',
            'feature_z': 'z_val_1'
        }
        score = self.cbrw.score(to_be_scored)
        self.assertFalse(isnan(score[0]))
        self.assertEqual(score[0], 0)

    def test_score_with_nans_default(self):
        obs = deepcopy(self.observations)
        obs[0]['feautre_a'] = np.nan

        to_be_scored = {
            'feature_a': np.nan,
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }

        # score observation with nan value
        self.cbrw.fit()
        score = self.cbrw.score(to_be_scored)
        self.assertTrue(isnan(score[0]))

        # fit includes observation with nan value
        self.cbrw = CBRW()
        self.cbrw.add_observations(obs)
        self.cbrw.fit()
        score = self.cbrw.score(to_be_scored)
        self.assertTrue(isnan(score[0]))

    def test_score_with_nans_ignore(self):
        obs = deepcopy(self.observations)
        obs[0]['feautre_a'] = np.nan

        to_be_scored = {
            'feature_a': np.nan,
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        actually_scored = {
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }

        # score observation with nan value
        self.cbrw = CBRW(ignore_unknown=True)
        self.cbrw.add_observations(self.observations)
        self.cbrw.fit()
        score = self.cbrw.score(to_be_scored)
        actual_score = self.cbrw.score(actually_scored)
        self.assertFalse(isnan(score[0]))
        self.assertEqual(score, actual_score)

        # fit includes observation with nan value
        self.cbrw = CBRW(ignore_unknown=True)
        self.cbrw.add_observations(obs)
        self.cbrw.fit()
        score = self.cbrw.score(to_be_scored)
        actual_score = self.cbrw.score(actually_scored)
        self.assertFalse(isnan(score[0]))
        self.assertEqual(score, actual_score)

    def test_value_scores_before_fit(self):
        with self.assertRaises(CBRWScoreError):
            _ = self.cbrw.value_scores(self.observations)

    def test_value_scores(self):
        self.cbrw.fit()

        # score observation where all features and values
        # have been previously observed
        to_be_scored = self.observations[0]
        value_scores = self.cbrw.value_scores(to_be_scored)
        value_scores = value_scores[0]
        self.assertListEqual(sorted(value_scores.keys()), sorted(to_be_scored.keys()))
        for vs in itervalues(value_scores):
            self.assertGreaterEqual(vs, 0)
            self.assertLessEqual(vs, 1)

    def test_value_scores_unknown_features_default(self):
        self.cbrw.fit()

        # score observation where all features but not all
        # values have been previously observed
        to_be_scored = {
            'feature_a': 'a_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        value_scores = self.cbrw.value_scores(to_be_scored)
        value_scores = value_scores[0]
        self.assertTrue(isnan(value_scores['feature_a']))
        self.assertFalse(isnan(value_scores['feature_b']))
        self.assertFalse(isnan(value_scores['feature_c']))

        # score observation where a feature has not
        # been previously observed
        to_be_scored = {
            'feature_x': 'x_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        value_scores = self.cbrw.value_scores(to_be_scored)
        value_scores = value_scores[0]
        self.assertTrue(isnan(value_scores['feature_x']))
        self.assertFalse(isnan(value_scores['feature_b']))
        self.assertFalse(isnan(value_scores['feature_c']))

        # score valid and invalid observations in one call
        to_be_scored = [
            self.observations[0],
            {'feature_x': 'x_val_x', 'feature_b': 'b_val_1'}
        ]
        value_scores = self.cbrw.value_scores(to_be_scored)
        valid_scores = value_scores[0]
        invalid_scores = value_scores[1]
        self.assertTrue(all(not isnan(valid_score) for valid_score in itervalues(valid_scores)))
        self.assertTrue(any(isnan(invalid_score) for invalid_score in itervalues(invalid_scores)))

    def test_value_scores_unknown_features_ignore(self):
        self.cbrw = CBRW(ignore_unknown=True)
        self.cbrw.add_observations(self.observations)
        self.cbrw.fit()

        # score observation where all features but not all
        # values have been previously observed
        to_be_scored = {
            'feature_a': 'a_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        actually_scored = {
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        value_scores = self.cbrw.value_scores(to_be_scored)[0]
        actual_value_scores = self.cbrw.value_scores(actually_scored)[0]
        self.assertTrue(all(not isnan(vs) for vs in itervalues(value_scores)))
        self.assertEqual(value_scores['feature_a'], 0)
        self.assertEqual(value_scores['feature_b'], actual_value_scores['feature_b'])
        self.assertEqual(value_scores['feature_c'], actual_value_scores['feature_c'])

        # score observation where a feature has not
        # been previously observed
        to_be_scored = {
            'feature_x': 'x_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        actually_scored = {
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        value_scores = self.cbrw.value_scores(to_be_scored)[0]
        actual_value_scores = self.cbrw.value_scores(actually_scored)[0]
        self.assertTrue(all(not isnan(vs) for vs in itervalues(value_scores)))
        self.assertEqual(value_scores['feature_x'], 0)
        self.assertEqual(value_scores['feature_b'], actual_value_scores['feature_b'])
        self.assertEqual(value_scores['feature_c'], actual_value_scores['feature_c'])

        # score observation where no features have
        # previously been observed
        to_be_scored = {
            'feature_x': 'x_val_x',
            'feature_y': 'y_val_1',
            'feature_z': 'z_val_1'
        }
        value_scores = self.cbrw.value_scores(to_be_scored)[0]
        self.assertTrue(all(not isnan(vs) for vs in itervalues(value_scores)))
        self.assertTrue(all(vs == 0 for vs in itervalues(value_scores)))

    def test_value_scores_with_nans_default(self):
        obs = deepcopy(self.observations)
        obs[0]['feautre_a'] = np.nan

        to_be_scored = {
            'feature_a': np.nan,
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }

        # score observation with nan value
        self.cbrw.fit()
        value_scores = self.cbrw.value_scores(to_be_scored)
        self.assertTrue(isnan(value_scores[0]['feature_a']))

        # fit includes observation with nan value
        self.cbrw = CBRW()
        self.cbrw.add_observations(obs)
        self.cbrw.fit()
        value_scores = self.cbrw.value_scores(to_be_scored)
        self.assertTrue(isnan(value_scores[0]['feature_a']))

    def test_value_scores_with_nans_ignore(self):
        obs = deepcopy(self.observations)
        obs[0]['feautre_a'] = np.nan

        to_be_scored = {
            'feature_a': np.nan,
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        actually_scored = {
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }

        # score observation with nan value
        self.cbrw = CBRW(ignore_unknown=True)
        self.cbrw.add_observations(self.observations)
        self.cbrw.fit()
        value_scores = self.cbrw.value_scores(to_be_scored)[0]
        actual_value_scores = self.cbrw.value_scores(actually_scored)[0]
        self.assertTrue(all(not isnan(vs) for vs in itervalues(value_scores)))
        self.assertEqual(value_scores['feature_a'], 0)
        self.assertEqual(value_scores['feature_b'], actual_value_scores['feature_b'])
        self.assertEqual(value_scores['feature_c'], actual_value_scores['feature_c'])

        # fit includes observation with nan value
        self.cbrw = CBRW(ignore_unknown=True)
        self.cbrw.add_observations(obs)
        self.cbrw.fit()
        value_scores = self.cbrw.value_scores(to_be_scored)[0]
        actual_value_scores = self.cbrw.value_scores(actually_scored)[0]
        self.assertTrue(all(not isnan(vs) for vs in itervalues(value_scores)))
        self.assertEqual(value_scores['feature_a'], 0)
        self.assertEqual(value_scores['feature_b'], actual_value_scores['feature_b'])
        self.assertEqual(value_scores['feature_c'], actual_value_scores['feature_c'])
