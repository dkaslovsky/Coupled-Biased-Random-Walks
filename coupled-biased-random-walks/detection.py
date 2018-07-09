from __future__ import division

from scipy.sparse import csr_matrix
from six import iteritems
from six.moves import zip

from count import ObservationCounter


class CBRW(object):

    def __init__(self):
        self.counter = ObservationCounter()
        self._prob_matrix = None
        self._bias_dict = None

    def add_observations(self, observation_iterable):
        self.counter.update(observation_iterable)

    def fit(self):
        if self.counter.n_obs == 0:
            raise ValueError('no observations provided')
        self._compute_biases()
        self._compute_prob_matrix()
        return self

    def _compute_prob_matrix(self):
        idx = []
        prob = []
        for (symbol1, symbol2), joint_count in iteritems(self.counter.joint_counts):

            # get index for symbols
            symb1_idx = self.counter.index[symbol1]
            symb2_idx = self.counter.index[symbol2]

            # get individual counts for symbols
            symb1_count = self.counter.get_count(symbol1)
            symb2_count = self.counter.get_count(symbol2)

            # p(symb1 | symb2)
            idx.append((symb1_idx, symb2_idx))
            prob.append(self._bias_dict[symbol2] * joint_count / symb2_count)
            # p(symb2 | symb1)
            idx.append((symb2_idx, symb1_idx))
            prob.append(self._bias_dict[symbol1] * joint_count / symb1_count)

        n_symb = len(self.counter.index)
        self._prob_matrix = csr_matrix((prob, zip(*idx)), shape=(n_symb, n_symb))

    def _compute_biases(self):
        bias_dict = {}
        for feature_name, value_counts in iteritems(self.counter.counts):
            mode = self._get_mode(value_counts)
            base = 1 - mode / self.counter.n_obs
            bias_dict.update({feature_val: self._compute_bias(count, mode, base)
                              for feature_val, count in iteritems(value_counts)})
        self._bias_dict = bias_dict

    @staticmethod
    def _compute_bias(count, mode, base):
        dev = 1 - count / mode
        return 0.5 * (dev + base)

    @staticmethod
    def _get_mode(counter):
        return counter.most_common(1)[0][1]
