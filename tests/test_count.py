import unittest
from collections import Counter

import numpy as np
from coupled_biased_random_walks.count import (IncrementingDict,
                                               ObservationCounter,
                                               get_feature_value, get_mode,
                                               isnan)
from six import iteritems


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
        {'feature_b': 'b_val_1', 'feature_c': 'c_val_2', 'feature_a': 'a_val_1'}
    ]

    # keep a set of all feature_name, feature_val pairs for testing
    all_index_keys = set()
    for observation in observations:
        for item in iteritems(observation):
            all_index_keys.add(item)

    def setUp(self):
        self.oc = ObservationCounter()
        self.oc.update(self.observations)

    def test_update(self):
        # test n_obs
        expected_counts = {
            'feature_a': 2,
            'feature_b': 2,
            'feature_c': 2
        }
        for feature_name, count in iteritems(self.oc.n_obs):
            self.assertEqual(count, expected_counts[feature_name])

        # test index
        self.assertSetEqual(set(self.oc.index.keys()), self.all_index_keys)

        # test counts
        # feature a
        feature_a_counts = self.oc.counts['feature_a']
        expected_a_items = [
            (('feature_a', 'a_val_1'), 2)
        ]
        self.assertListEqual(list(feature_a_counts.items()), expected_a_items)
        # feature b
        feature_b_counts = self.oc.counts['feature_b']
        expected_b_items = [
            (('feature_b', 'b_val_1'), 2)
        ]
        self.assertListEqual(list(feature_b_counts.items()), expected_b_items)
        # feature c
        feature_c_counts = self.oc.counts['feature_c']
        expected_c_items = [
            (('feature_c', 'c_val_1'), 1),
            (('feature_c', 'c_val_2'), 1)
        ]
        self.assertListEqual(list(feature_c_counts.items()), expected_c_items)

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


class TestObservationCounterWithMissingData(unittest.TestCase):
    """
    Unit tests for ObservationCounter
    """

    observations = [
        {'feature_a': 'a_val_1', 'feature_c': 'c_val_2', 'feature_d': np.nan},
        {'feature_b': 'b_val_1', 'feature_a': 'a_val_1', 'feature_c': np.nan}
    ]

    # keep a set of all feature_name, feature_val pairs for testing
    all_index_keys = set()
    for observation in observations:
        for item in iteritems(observation):
            if not isnan(get_feature_value(item)):
                all_index_keys.add(item)

    def setUp(self):
        self.oc = ObservationCounter()
        self.oc.update(self.observations)

    def test_update(self):
        # test n_obs
        expected_counts = {
            'feature_a': 2,
            'feature_b': 1,
            'feature_c': 1
        }
        for feature_name, count in iteritems(self.oc.n_obs):
            self.assertEqual(count, expected_counts[feature_name])

        # test index
        self.assertSetEqual(set(self.oc.index.keys()), self.all_index_keys)

        # test counts
        # feature a
        feature_a_counts = self.oc.counts['feature_a']
        expected_a_items = [
            (('feature_a', 'a_val_1'), 2)
        ]
        self.assertListEqual(list(feature_a_counts.items()), expected_a_items)
        # feature b
        feature_b_counts = self.oc.counts['feature_b']
        expected_b_items = [
            (('feature_b', 'b_val_1'), 1)
        ]
        self.assertListEqual(list(feature_b_counts.items()), expected_b_items)
        # feature c
        feature_c_counts = self.oc.counts['feature_c']
        expected_c_items = [
            (('feature_c', 'c_val_2'), 1)
        ]
        self.assertListEqual(list(feature_c_counts.items()), expected_c_items)

        # test joint_counts
        expected_joint_counts = {
            (('feature_a', 'a_val_1'), ('feature_b', 'b_val_1')): 1,
            (('feature_a', 'a_val_1'), ('feature_c', 'c_val_2')): 1,
        }
        self.assertDictEqual(self.oc.joint_counts, expected_joint_counts)


class TestIsNaN(unittest.TestCase):
    """
    Unit tests for isnan()
    """

    def test_isnan(self):
        table = {
            'numpy nan': {
                'test': np.nan,
                'expected': True
            },
            'float nan': {
                'test': float('nan'),
                'expected': True
            },
            'int zero': {
                'test': 0,
                'expected': False
            },
            'float zero': {
                'test': 0.0,
                'expected': False
            },
            'int nonzero': {
                'test': 456,
                'expected': False
            },
            'float nonzero': {
                'test': 10.123,
                'expected': False
            },
            'string': {
                'test': 'nan',
                'expected': False
            },
        }
        for test_name, test in iteritems(table):
            isnan_result = isnan(test['test'])
            self.assertEqual(isnan_result, test['expected'], test_name)


class TestGetMode(unittest.TestCase):
    """
    Unit tests for get_mode()
    """

    def setUp(self):
        self.c1 = Counter()
        self.c1.update(['a', 'a', 'a', 'b', 'b'])
        self.c2 = Counter()
        self.c2.update(['a', 'a', 'b', 'b'])
    
    def test_get_mode(self):
        table = {
            'empty counter': {
                'counter': Counter(),
                'expected': 0
            },
            'unique mode': { 
                'counter': self.c1,
                'expected': 3
            },
            'nonunique mode': {
                'counter': self.c2,
                'expected': 2
            },
        }
        for test_name, test in iteritems(table):
            mode = get_mode(test['counter'])
            self.assertEqual(mode, test['expected'], test_name)
