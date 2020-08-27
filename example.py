import os

from coupled_biased_random_walks import CBRW
from data.loading import load_from_csv


file_dir = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = os.path.join(file_dir, 'data', 'CBRW_paper_example.csv')
EXCLUDE_COLS = ['Cheat?']


def round_dict_values(input_dict, digits=4):
    """ Helper function for printing dicts with float values """
    return {key: round(val, digits) for key, val in input_dict.items()}


if __name__ == '__main__':

    detector = CBRW()

    # load data and add to detector as observations
    observations = load_from_csv(DATA_PATH, exclude_cols=EXCLUDE_COLS)

    # add observations to detector and fit
    detector.add_observations(observations)
    detector.fit()

    # compute scores
    scores = detector.score(observations)
    value_scores = detector.value_scores(observations)

    # display results
    print(f'Detector fit with {len(observations)} observations:')
    for i, obs in enumerate(observations):
        print(f'Observation ID {i+1}: {obs}')

    print('\nFeature weights:')
    print(round_dict_values(detector.feature_weights, 4))

    print('\nScores:')
    for i, score in enumerate(scores):
        print(f'Observation ID {i+1}: {round(score, 4)}')

    print('\nValue scores per attribute:')
    for i, value_score in enumerate(value_scores):
        print(f'Observation ID {i+1}: {round_dict_values(value_score, 4)}')
