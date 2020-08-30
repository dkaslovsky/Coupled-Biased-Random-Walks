from typing import Dict, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from coupled_biased_random_walks.types import obs_item_type


EPS = 10e-8  # tolerance below which to consider a value as zero


def random_walk(
    transition_matrix: csr_matrix,
    alpha: float,
    err_tol: float,
    max_iter: int
) -> np.ndarray:
    """
    Run random walk to compute stationary probabilities
    :param transition_matrix: scipy.sparse.csr_matrix defining the random walk
    :param alpha: damping parameter
    :param err_tol: convergence criterion for stationary probability
    :param max_iter: max number of steps to take for random walk
    :return:
    """
    # shape of transition matrix will be length of vectors
    n = transition_matrix.shape[0]
    # damping vector
    damping_vec = ((1 - alpha) / n) * np.ones((n, 1))
    # stationary vector initialization
    pi = (1 / n) * np.ones((n, 1))

    for _ in range(max_iter):
        pi_next = damping_vec + alpha * transition_matrix.T.dot(pi)
        err = np.linalg.norm(pi - pi_next, ord=np.inf)
        pi = pi_next
        if err <= err_tol:
            break

    pi_sum = sum(pi)
    if pi_sum < EPS:
        raise ValueError('stationary probabilities sum approximately zero')
    return pi / pi_sum


def dict_to_csr_matrix(
    data_dict: Dict[obs_item_type, float],
    shape: Union[int, Tuple[int, int]],
) -> csr_matrix:
    """
    Converts dict of index -> value to csr_matrix
    :param data_dict: dict mapping matrix index tuple to corresponding matrix value
    :param shape: (row, col) tuple for shape of csr_matrix (also accepts int when row = col)
    """
    if not data_dict:
        raise ValueError('dict must not be empty')

    if isinstance(shape, int):
        shape = (shape, shape)
    data = list(data_dict.values())  # csr_matrix cannot accept iterator for data
    idx = zip(*list(data_dict.keys()))
    return csr_matrix((data, idx), shape=shape)


def row_normalize_csr_matrix(matrix: csr_matrix) -> csr_matrix:
    """
    Row normalize a csr matrix without mutating the input
    :param matrix: scipy.sparse.csr_matrix instance
    """
    if not isinstance(matrix, csr_matrix):
        raise TypeError('expected input to be a scipy csr_matrix')
    if any(matrix.data == 0):
        raise ValueError('input must be scipy.sparse.csr_matrix and must not store zeros')
    # get row index for every nonzero element in matrix
    row_idx, col_idx = matrix.nonzero()
    # compute unraveled row sums
    row_sums = matrix.sum(axis=1).A1
    # divide data by (broadcasted) row sums
    normalized = matrix.data / row_sums[row_idx]
    return csr_matrix((normalized, (row_idx, col_idx)), shape=matrix.shape)
