import numpy as np

from coupled_biased_random_walks.detection import CBRW
from data.loading import load_from_csv


DATA_PATH = './data/CBRW_paper_example.csv'
EXCLUDE_COLS = ['Cheat?']


if __name__ == '__main__':

    detector = CBRW()

    # load data and add to detector as observations
    data = load_from_csv(DATA_PATH, exclude_cols=EXCLUDE_COLS)
    detector.add_observations(data)

    # fit and score data
    detector.fit()
    scores = detector.score(data)

    # print scores in descending order
    idx = np.argsort(scores)[::-1]
    for i in idx:
        print('Score: {} | Data: {}'.format(round(scores[i], 4), data[i]))
