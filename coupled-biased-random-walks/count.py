from collections import Counter, defaultdict
from itertools import combinations, tee

from six import iteritems


class FixedValueDict(dict):
    def __setitem__(self, key, value):
        if dict.has_key(self, key):
            raise KeyError('cannot overwrite key {}'.format(key))
        dict.__setitem__(self, key, value)


class IncrementingDict(object):
    def __init__(self):
        self._next_val = 0
        self._d = FixedValueDict()

    def insert(self, key):
        try:
            self._d[key] = self._next_val
        except KeyError:
            return
        self._next_val += 1

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __getitem__(self, key):
        return self._d.__getitem__(key)

    def __len__(self):
        return self._d.__len__()

    def __repr__(self):
        return self._d.__repr__()


class ObservationCounter(object):
    def __init__(self):
        self.n_obs = 0
        self._counts = defaultdict(Counter)
        self._joint_counts = Counter()
        self._index = IncrementingDict()

    @property
    def counts(self):
        return dict(self._counts)

    @property
    def joint_counts(self):
        return dict(self._joint_counts)

    @property
    def index(self):
        return self._index

    def update(self, observation_iterable):
        """

        :param observation_iterable: list of dicts
        """
        if isinstance(observation_iterable, dict):
            observation_iterable = [observation_iterable]
        for observation in observation_iterable:
            if not isinstance(observation, dict):
                raise TypeError('observation must be a dict')
            self._update(observation)

    def _update(self, observation):
        obs1, obs2, obs3 = tee(iteritems(observation), 3)
        self._update_counts(obs1)
        self._update_joint_counts(obs2)
        self._update_index(obs3)
        # in the future we might need to track n_obs per feature
        # and store it in a dict; this might require skipping
        # features with value nan
        self.n_obs += 1

    def _update_counts(self, observation):
        for item in observation:
            feature_name = self._get_feature_name(item)
            self._counts[feature_name].update([item])

    def _update_joint_counts(self, observation):
        pairs = combinations(sorted(observation), 2)
        self._joint_counts.update(pairs)

    def _update_index(self, observation):
        for item in observation:
            self._index.insert(item)

    @staticmethod
    def _get_feature_name(feature_tuple):
        return feature_tuple[0]

    def get_count(self, item):
        feature_name = self._get_feature_name(item)
        try:
            return self._counts.get(feature_name)[item]
        except TypeError:
            # feature_name is not in self._counts
            return 0
