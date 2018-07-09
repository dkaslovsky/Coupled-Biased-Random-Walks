from string import ascii_uppercase

import numpy as np
import pandas as pd
from six import iteritems

np.random.seed(0)


OBS_LENS = [2, 3, 4]

FEATURES = {
    'feature1': list(ascii_uppercase[:10]),
    'feature2': list(ascii_uppercase[10:13]),
    'feature3': list(ascii_uppercase[13:17]),
}

def generate_observations(n_obs, vocab_len=26):
    vocab = list(ascii_uppercase[:vocab_len])
    return [np.random.choice(vocab, np.random.choice(OBS_LENS), replace=True).tolist()
            for _ in range(n_obs)]


def generate_df(n_obs):
    data = {feature_name: np.random.choice(values, n_obs, replace=True)
            for feature_name, values in iteritems(FEATURES)}
    return pd.DataFrame(data)
