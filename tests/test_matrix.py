import unittest

import numpy as np
from scipy.sparse import csr_matrix

from coupled_biased_random_walks.matrix import (random_walk,
                                                row_normalize_csr_matrix)


def construct_2x2_matrix(data):
    idx = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return csr_matrix((data, zip(*idx)), shape=(2, 2))


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

    def test_row_normalize(self):
        for _ in range(10):
            data = np.random.rand(4)
            matrix = construct_2x2_matrix(data)
            matrix = row_normalize_csr_matrix(matrix)
            sums = matrix.sum(axis=1)
            self.assertAlmostEqual(sums[0], 1, 3)
            self.assertAlmostEqual(sums[1], 1, 3)
