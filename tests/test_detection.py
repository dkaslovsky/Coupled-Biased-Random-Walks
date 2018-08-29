import unittest

from scipy.sparse import csr_matrix

from coupled_biased_random_walks.detection import CBRW


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

    def test_get_node_score(self):
        self.cbrw._stationary_prob = {
            ('feature_a', 'a_val_1'): 0.25
        }
        # get score for valid node
        valid_node_score = self.cbrw._get_node_score(('feature_a', 'a_val_1'))
        self.assertEqual(valid_node_score, 0.25)
        # get score for invalid node - should raise exception
        with self.assertRaises(ValueError):
            _ = self.cbrw._get_node_score(('feature_a', 'xxx'))

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
        bias_dict = self.cbrw._compute_biases()
        transition_matrix = self.cbrw._compute_biased_transition_matrix(bias_dict)
        self.assertIsInstance(transition_matrix, csr_matrix)
        self.assertTupleEqual(transition_matrix.shape, (4, 4))
        self.assertTrue((0 < transition_matrix.data).all())
        self.assertTrue((transition_matrix.data <= 1).all())

    def test_fit_no_data(self):
        cbrw = CBRW()
        with self.assertRaises(ValueError):
            cbrw.fit()
    
    def test_fit(self):
        self.cbrw.fit()
        self.assertIsNotNone(self.cbrw._stationary_prob)
        self.assertIsNotNone(self.cbrw._feature_relevance)

    def test_score(self):
        self.cbrw.fit()

        # score observation where all features and values
        # have been previously observed
        to_be_scored = self.observations[0]
        score = self.cbrw.score(to_be_scored)
        score = score[0]
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

        # score observation where all features but not all
        # values have been previously observed
        to_be_scored = {
            'feature_a': 'a_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        with self.assertRaises(ValueError):
            _ = self.cbrw.score(to_be_scored)

        # score observation where a feature has not
        # been previously observed
        to_be_scored = {
            'feature_x': 'x_val_x',
            'feature_b': 'b_val_1',
            'feature_c': 'c_val_1'
        }
        with self.assertRaises(ValueError):
            _ = self.cbrw.score(to_be_scored)
