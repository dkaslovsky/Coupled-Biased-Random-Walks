from string import ascii_uppercase

import numpy as np
np.random.seed(0)


OBS_LENS = [2, 3, 4]

def generate_observations(n_obs, vocab_len=26):
    vocab = list(ascii_uppercase[:vocab_len])
    return [np.random.choice(vocab, np.random.choice(OBS_LENS), replace=True).tolist()
            for _ in range(n_obs)]
