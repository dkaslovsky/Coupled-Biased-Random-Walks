import unittest
from typing import List

import numpy as np
from scipy.sparse import csr_matrix

from coupled_biased_random_walks.matrix import (
    dict_to_csr_matrix,
    random_walk,
    row_normalize_csr_matrix,
)

np.random.seed(0)


def construct_2x2_csr_matrix(data: List[float]) -> csr_matrix:
    """
    Construct a 2x2 csr_matrix
    :param data: list of length 4 of data for csr matrix corresponding to idx position
    """
    idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    matrix_data = []
    matrix_idx = []
    for ix, datum in zip(idx, data):
        if datum == 0:
            continue
        matrix_data.append(datum)
        matrix_idx.append(ix)
    if not matrix_data:
        return csr_matrix(([], ([], [])), shape=(2, 2))
    return csr_matrix((matrix_data, zip(*matrix_idx)), shape=(2, 2))


def csr_matrix_equality(c1: csr_matrix, c2: csr_matrix) -> bool:
    """
    Test two csr matrices for equality
    :param c1: csr_matrix to compare
    :param c2: csr_matrix to compare
    """
    if c1.shape != c2.shape:
        return False
    # more efficient to test elements for inequality
    return (c1 != c2).nnz == 0


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
        matrix = construct_2x2_csr_matrix(data)
        pi = random_walk(matrix, alpha=self.alpha, err_tol=self.err_tol, max_iter=self.max_iter)
        self.assertEqual(len(pi), 2)
        self.assertAlmostEqual(pi[0], 0.5, 3)
        self.assertAlmostEqual(pi[1], 0.5, 3)

        # prob 1, 0 (alpha = 1)
        data = [1, 0, 1, 0]
        matrix = construct_2x2_csr_matrix(data)
        pi = random_walk(matrix, alpha=1, err_tol=self.err_tol, max_iter=self.max_iter)
        self.assertEqual(len(pi), 2)
        self.assertAlmostEqual(pi[0], 1, 3)
        self.assertAlmostEqual(pi[1], 0, 3)


class TestDictToCSRMatrix(unittest.TestCase):
    """
    Unit tests for dict_to_csr_matrix
    """

    def test_dict_to_csr_matrix(self):
        table = {
            'test 1': {
                'data_dict': {(0, 1): 25, (1, 0): 16},
                'shape': 2,
                'expected': construct_2x2_csr_matrix([0, 25, 16, 0])
            },
            'test 2': {
                'data_dict': {(0, 0): 1, (1, 1): 1},
                'shape': 2,
                'expected': construct_2x2_csr_matrix([1, 0, 0, 1])
            }
        }

        for test_name, test in table.items():
            result = dict_to_csr_matrix(test['data_dict'], test['shape'])
            self.assertTrue(csr_matrix_equality(result, test['expected']), test_name)


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
                'input': np.array([[1, 2], [3, 4]]),
                'exception': TypeError
            },
            'stored zeros': {
                'input': csr_matrix(([1, 0], ([0, 1], [0, 0])), shape=(2, 2)),
                'exception': ValueError
            }
        }

        for test_name, test in valid_table.items():
            matrix = construct_2x2_csr_matrix(test['data'])
            normalized = row_normalize_csr_matrix(matrix)
            row_sums = normalized.sum(axis=1)
            self.assertAlmostEqual(row_sums[0], test['expected_row_0'], 3, test_name)
            self.assertAlmostEqual(row_sums[1], test['expected_row_1'], 3, test_name)

        for test_name, test in invalid_table.items():
            with self.assertRaises(test['exception']):
                _ = row_normalize_csr_matrix(test['input'])
