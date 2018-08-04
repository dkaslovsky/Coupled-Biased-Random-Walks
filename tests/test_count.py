import unittest

from six import iteritems

from coupled_biased_random_walks.count import (IncrementingDict,
                                               ObservationCounter)


class TestIncrementingDict(unittest.TestCase):
    """
    Unit tests for IncrementingDict
    """

    def setUp(self):
        self.d = IncrementingDict()

    def test_insert(self):
        table = {
            'insert unique keys': {
                'keys_to_insert': ['a', 'c', 'b', 'd'],
                'expected_dict': {'a': 0, 'c': 1, 'b': 2, 'd': 3}
            },
            'insert repeated keys': {
                'keys_to_insert': ['a', 'c', 'b', 'c', 'a', 'b', 'd'],
                'expected_dict': {'a': 0, 'c': 1, 'b': 2, 'd': 3}
            }
        }
        for test_name, test in iteritems(table):
            self.setUp()
            for key in test['keys_to_insert']:
                self.d.insert(key)
            self.assertEqual(self.d, test['expected_dict'], test_name)

    def test_no_setitem(self):
        with self.assertRaises(TypeError):
            self.d['a'] = 0

    def test_getitem(self):
        self.d.insert('a')
        val = self.d['a']
        self.assertEqual(val, 0)

    def test_len(self):
        self.d.insert('a')
        self.d.insert('a')
        self.d.insert('b')
        self.assertEqual(len(self.d), 2)


class TestObservationCounter(unittest.TestCase):
    """
    Unit tests for ObservationCounter
    """

    observations = [
        {'feature_a': 'a_val_1', 'feature_b': 'b_val_1', 'feature_c': 'c_val_1'},
        {'feature_b': 'b_val_1', 'feature_c': 'c_val_2', 'feature_a': 'a_val_1',}
    ]
    all_keys = set(list(observations[0].items()) + list(observations[1].items()))

    def setUp(self):
        self.oc = ObservationCounter()
        self.oc.update(self.observations)

    def test_update(self):
        # test n_obs
        self.assertEqual(self.oc.n_obs, 2)

        # test index
        self.assertSetEqual(set(self.oc.index.keys()), self.all_keys)

        # test counts
        # feature a
        feature_a_counts = self.oc.counts['feature_a']
        expected_a_items = [
            (('feature_a', 'a_val_1'), 2)
        ]
        self.assertListEqual(feature_a_counts.items(), expected_a_items)
        # feature b
        feature_b_counts = self.oc.counts['feature_b']
        expected_b_items = [
            (('feature_b', 'b_val_1'), 2)
        ]
        self.assertListEqual(feature_b_counts.items(), expected_b_items)
        # feature c
        feature_c_counts = self.oc.counts['feature_c']
        expected_c_items = [
            (('feature_c', 'c_val_1'), 1),
            (('feature_c', 'c_val_2'), 1)
        ]
        self.assertListEqual(feature_c_counts.items(), expected_c_items)

        # test joint_counts
        expected_joint_counts = {
            (('feature_a', 'a_val_1'), ('feature_b', 'b_val_1')): 2,
            (('feature_a', 'a_val_1'), ('feature_c', 'c_val_1')): 1,
            (('feature_a', 'a_val_1'), ('feature_c', 'c_val_2')): 1,
            (('feature_b', 'b_val_1'), ('feature_c', 'c_val_1')): 1,
            (('feature_b', 'b_val_1'), ('feature_c', 'c_val_2')): 1,
        }
        self.assertDictEqual(self.oc.joint_counts, expected_joint_counts)


    def test_get_count(self):
        table = {
            'exists count 1': {
                'feature tuple': ('feature_c', 'c_val_2'),
                'expected': 1
            },
            'exists count 2': {
                'feature tuple': ('feature_b', 'b_val_1'),
                'expected': 2
            },
            'feature does not exist': {
                'feature tuple': ('feature_z', 'z_val_1'),
                'expected': 0
            },
            'val does not exist': {
                'feature tuple': ('feature_c', 'c_val_3'),
                'expected': 0
            },
        }
        for test_name, test in iteritems(table):
            count = self.oc.get_count(test['feature tuple'])
            expected = test['expected']
            self.assertEqual(count, expected, test_name)
