import unittest

from coupled_biased_random_walks.detection import CBRW


# TODO: how can rows in the trans matrix be all zero?


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
        self.cbrw.fit()

    def tearDown(self):
        self.cbrw._stationary_prob = None
        self.cbrw._feature_relevance = None

    def test_fit(self):
        pass

    def test_score(self):
        pass

    def test_get_node_score(self):
        pass

    def test_get_feature_relevance(self):
        rel = self.cbrw._get_feature_relevance(('feature_a', 'a_val_1'))
        self.assertGreaterEqual(rel, 0)
        self.assertLessEqual(rel, 1)

        rel = self.cbrw._get_feature_relevance(('feature_a', 'xxx'))
        self.assertEqual(rel, 0)

        rel = self.cbrw._get_feature_relevance(('xxx', 'xxx'))
        self.assertEqual(rel, 0)


    def test_compute_biased_transition_matrix(self):
        pass

    def test_compute_biases(self):
        pass