from six.moves import zip

from coupled_biased_random_walks import CBRW
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

    # print scores and observations
    for score, datum in zip(scores, data):
        print('Score: {} | Data: {}'.format(round(score, 4), datum))
