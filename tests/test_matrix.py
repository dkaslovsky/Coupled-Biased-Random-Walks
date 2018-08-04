import unittest

import numpy as np
from scipy.sparse import csr_matrix
from six import iteritems

from coupled_biased_random_walks.matrix import (random_walk,
                                                row_normalize_csr_matrix)

np.random.seed(0)


def construct_2x2_matrix(data):
    idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    matrix_data = []
    matrix_idx = []
    for i, d in enumerate(data):
        if d != 0:
            matrix_data.append(d)
            matrix_idx.append(idx[i])
    if matrix_data:
        return csr_matrix((matrix_data, zip(*matrix_idx)), shape=(2, 2))
    return csr_matrix(([], ([], [])), shape=(2 ,2))


class TestRandomWalk(unittest.TestCase):
    """
    Unit tests for random_walk
    """

    alpha = 0.95
    err_tol = 1e-3
    max_iter = 100

    def test_random_walk(self):
        # prob 0.5, 0.5
        data = [0, 1, 1, 0]
        matrix = construct_2x2_matrix(data)
        pi = random_walk(matrix, alpha=self.alpha, err_tol=self.err_tol, max_iter=self.max_iter)
        self.assertEqual(len(pi), 2)
        self.assertAlmostEqual(pi[0], 0.5, 3)
        self.assertAlmostEqual(pi[1], 0.5, 3)

        # prob 1, 0 (alpha = 1)
        data = [1, 0, 1, 0]
        matrix = construct_2x2_matrix(data)
        pi = random_walk(matrix, alpha=1, err_tol=self.err_tol, max_iter=self.max_iter)
        self.assertEqual(len(pi), 2)
        self.assertAlmostEqual(pi[0], 1, 3)
        self.assertAlmostEqual(pi[1], 0, 3)


class TestRowNormalizeCSRMatrix(unittest.TestCase):
    """
    Unit tests for row_normalize_csr_matrix
    """

    def test_valid_row_normalize(self):
        valid_table = {
            'random entries': {
                'data': np.random.rand(4),
                'expected_row_0': 1,
                'expected_row_1': 1,
            },
            'zero row': {
                'data': np.array([0, 0, 1, 1]),
                'expected_row_0': 0,
                'expected_row_1': 1,
            },
            'all zeros': {
                'data': np.zeros(4),
                'expected_row_0': 0,
                'expected_row_1': 0,
            }
        }

        invalid_table = {
            'wrong type': {
                'input': np.array([[1,2],[3,4]]),
                'exception': TypeError
            },
            'stored zeros': {
                'input': csr_matrix(([1, 0], ([0, 1], [0, 0])), shape=(2, 2)),
                'exception': ValueError
            }
        }

        for test_name, test in iteritems(valid_table):
            matrix = construct_2x2_matrix(test['data'])
            normalized = row_normalize_csr_matrix(matrix)
            row_sums = normalized.sum(axis=1)
            self.assertAlmostEqual(row_sums[0], test['expected_row_0'], 3, test_name)
            self.assertAlmostEqual(row_sums[1], test['expected_row_1'], 3, test_name)

        for test_name, test in iteritems(invalid_table):
            with self.assertRaises(test['exception']):
                _ = row_normalize_csr_matrix(test['input'])
