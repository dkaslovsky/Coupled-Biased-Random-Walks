from collections import Counter, defaultdict
from itertools import combinations, tee

from six import iteritems

try:
    # python 2
    from collections import Mapping
except ImportError:
    # python 3
    from collections.abc import Mapping


class IncrementingDict(Mapping):

    """
    Dict-like class for assigning an incrementing value to new keys and
    does not allow overwriting of a key.
    Inherits abstract base class Mapping instead of dict; note that we
    intentionally do not define a __setitem__ method
    """

    def __init__(self):
        self._d = {}
        self._next_val = 0

    def insert(self, key):
        """
        Inserts a (strictly new) key
        :param key: any hashable object to be used as a key
        """
        if key in self._d:
            return
        self._d[key] = self._next_val
        self._next_val += 1

    def __getitem__(self, key):
        return self._d[key]

    def __iter__(self):
        return self._d.__iter__()

    def __len__(self):
        return self._d.__len__()

    def __repr__(self):
        return self._d.__repr__()


class ObservationCounter(object):

    """
    Counts single and joint occurrences of key/value pairs in a dict with
    the intention that an observation of categorical features is represented
    as a dict of {feature_name: feature_value, ...}
    """

    def __init__(self):
        # stores count of observations for each feature name; somewhat redundant
        # with self._counts but eliminates the need to iterate and sum
        self.n_obs = Counter()
        # nested dict storing individual counts of features, keyed by feature name
        # and then by (feature_name, feature_value) tuple
        self._counts = defaultdict(Counter)
        # stores joint counts of features, keyed by (feature_tuple1, feature_tuple2)
        # where each feature tuple takes the form (feature_name, feature_value)
        self._joint_counts = Counter()
        # maps each feature tuple to a (unique, incrementing integer) index
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
        Update counts with new observation(s)
        :param observation_iterable: list of dicts
        """
        if isinstance(observation_iterable, dict):
            observation_iterable = [observation_iterable]
        for observation in observation_iterable:
            # feature name with value NaN represents a missing feature in the
            # observation (e.g., a missing value is NaN-filled in a pandas DataFrame) so
            # we remove any such features from the observation to avoid including in counts
            obs = {key: value for key, value in iteritems(observation) if not isnan(value)}
            # create iterators of obs for updating counts
            obs1, obs2 = tee(iteritems(obs), 2)
            self._update_counts(obs1)
            self._update_joint_counts(obs2)

    def get_count(self, item):
        """
        Getter to safely retrieve count from interal data structure of defaultdict(Counter)
        :param item: tuple of the form ('feature_name', 'feature_value')
        """
        feature_name = get_feature_name(item)
        feature_counts = self._counts.get(feature_name)
        try:
            return feature_counts.get(item, 0)
        except AttributeError:
            # AttributeError raised when feature_counts is None
            # meaning there is no count for the feature_name
            return 0
    
    def _update_counts(self, observation):
        """
        Update single counts
        :param observation: iterable of tuples of the form ('feature_name', 'feature_value')
        """
        for item in observation:
            feature_name = get_feature_name(item)
            self._counts[feature_name].update([item])
            self._index.insert(item)
            self.n_obs.update([feature_name])

    def _update_joint_counts(self, observation):
        """
        Update joint counts
        :param observation: iterable of tuples of the form ('feature_name', 'feature_value')
        """
        pairs = combinations(sorted(observation), 2)
        self._joint_counts.update(pairs)


# Helper functions

def get_feature_name(feature_tuple):
    """
    Helper function to return feature name from tuple representation
    :param feature_tuple: tuple of the form (feature_name, feature_value)
    """
    return feature_tuple[0]


def get_feature_value(feature_tuple):
    """
    Helper function to return feature value from tuple representation
    :param feature_tuple: tuple of the form (feature_name, feature_value)
    """
    return feature_tuple[1]


def get_mode(counter):
    """
    Helper function to return the count of the most common
    element from an instance of Counter()
    :param counter: collections.Counter instance
    """
    mode = counter.most_common(1)
    if not mode:
        return 0
    # if mode is not empty it will be a list containing a
    # single tuple from which we want the second element
    return mode[0][1]


def isnan(x):
    """
    Return True if x is NaN where x can be of any type
    :param x: any object for which (in)equality can be checked
    """
    return x != x
