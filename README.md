# Coupled-Biased-Random-Walks
Outlier detection for categorical data

[![Build Status](https://travis-ci.org/dkaslovsky/Coupled-Biased-Random-Walks.svg?branch=master)](https://travis-ci.org/dkaslovsky/Coupled-Biased-Random-Walks)
[![Coverage Status](https://coveralls.io/repos/github/dkaslovsky/Coupled-Biased-Random-Walks/badge.svg?branch=master)](https://coveralls.io/github/dkaslovsky/Coupled-Biased-Random-Walks?branch=master)

### Overview
Python [2.7, 3.x] implementation of the Coupled Biased Random Walks (CBRW) outlier detection algorithm described by Pang, Cao, and Chen in https://www.ijcai.org/Proceedings/16/Papers/272.pdf.

This implementation operates on Python dicts rather than Pandas DataFrames.  This has the advantage of allowing the model to be updated with new observations in a trivial manner and is more efficient in certain aspects.  However, these advantages come at the cost of iterating a (potentially large) dict of observed values more times than might otherwise be necessary using an underlying DataFrame implementation.

If one is working with data previously loaded into a DataFrame, simply use the result of `pandas.DataFrame.to_dict(orient='records')` instead of the DataFrame itself to add observations to the model.  Note that because it is common for a DataFrame to fill missing values with `nan`, the detector will ignore features with value `nan` in any observation record.  Therefore, there is no need to further preprocess the DataFrame before using its `to_dict` method to create records.

### Installation
This package is hosted on PyPI and can be installed via `pip`:
```
$ pip install coupled_biased_random_walks
```
To instead install from source:
```
$ git clone git@github.com:dkaslovsky/Coupled-Biased-Random-Walks.git
$ cd Coupled-Biased-Random-Walks
$ python setup.py install
```

### Example
Let's run the CBRW detection algorithm on the authors' example data set from the paper:

<img src="./example_table.png" width="400">

This data is saved as a [CSV file](./data/CBRW_paper_example.csv) in this repository and is loaded into memory as a list of dicts by [example.py](./example.py).  Note that we drop the `Cheat?` column when loading the data, as this is essentially the target variable indicating the anomalous activity to be detected.  The detector is instantiated and observations are added as follows:
```
>>> detector = CBRW()
>>> detector.add_observations(observations)
```
where `observations` is an iterable of dicts such as the one loaded from the example .CSV file.  Once all of the observations are loaded, the detector can be finalized for scoring by calling `fit()` and observations can then be scored.
```
>>> detector.fit()
>>> scores = detector.score(observations)
```
Even after fitting and scoring, more observations can be added via `add_observations` and the detector can again be fit to be used for scoring.  The advantage of this implementation is this ability to incrementally update with new observations.

The results of scoring the example data are shown below.  Note that the only observation (`ID=1`) where fraud was present (`Cheat? = yes`) received the largest anomaly score.
```
Scores:
Observation ID 1: 0.1055
Observation ID 2: 0.0797
Observation ID 3: 0.0741
Observation ID 4: 0.0805
Observation ID 5: 0.0992
Observation ID 6: 0.0752
Observation ID 7: 0.0741
Observation ID 8: 0.0815
Observation ID 9: 0.0728
Observation ID 10: 0.0979
Observation ID 11: 0.0812
Observation ID 12: 0.0887
```

The "value scores" (scores per attribute) for each observation can also be calculated
```
>>> value_scores(observations)
```
and the results for the example data are shown below.
```
Value scores:
Observation ID 1: {'Gender': 0.0088, 'Education': 0.0195, 'Marriage': 0.0379, 'Income': 0.0393}
Observation ID 2: {'Gender': 0.0171, 'Education': 0.0195, 'Marriage': 0.0208, 'Income': 0.0223}
Observation ID 3: {'Gender': 0.0088, 'Education': 0.0195, 'Marriage': 0.0212, 'Income': 0.0247}
Observation ID 4: {'Gender': 0.0088, 'Education': 0.0286, 'Marriage': 0.0208, 'Income': 0.0223}
Observation ID 5: {'Gender': 0.0171, 'Education': 0.0195, 'Marriage': 0.0379, 'Income': 0.0247}
Observation ID 6: {'Gender': 0.0088, 'Education': 0.0209, 'Marriage': 0.0208, 'Income': 0.0247}
Observation ID 7: {'Gender': 0.0088, 'Education': 0.0195, 'Marriage': 0.0212, 'Income': 0.0247}
Observation ID 8: {'Gender': 0.0171, 'Education': 0.0209, 'Marriage': 0.0212, 'Income': 0.0223}
Observation ID 9: {'Gender': 0.0088, 'Education': 0.0209, 'Marriage': 0.0208, 'Income': 0.0223}
Observation ID 10: {'Gender': 0.0088, 'Education': 0.0286, 'Marriage': 0.0212, 'Income': 0.0393}
Observation ID 11: {'Gender': 0.0171, 'Education': 0.0209, 'Marriage': 0.0208, 'Income': 0.0223}
Observation ID 12: {'Gender': 0.0088, 'Education': 0.0195, 'Marriage': 0.0212, 'Income': 0.0393}
```

The entire example can be reproduced by running:
```
$ python example.py
```

The CBRW algorithm can also be used to calculate feature weights.  These weights are calculated when the detector is fit and are used during scoring, but can also be used by any other outlier detection algorithm.  Thus, the CBRW algorithm can be used simply to calculate feature weights and need not score observations.  Feature weights are stored as a property of the detector after the detector's `fit` method has been called:
```
>>> detector = CBRW()
>>> detector.add_observations(observations)
>>> detector.fit()
>>> detector.feature_weights
```
For the example data, the computed feature weights are
```
Feature weights:
{'Gender': 0.1608, 'Education': 0.2627, 'Marriage': 0.2826, 'Income': 0.2939}
```

### Implementation Notes
- For efficiency, the detector state is only (re)computed upon calling `.fit()`.  Therefore adding new observations (`.add_observations()`) will not affect scoring until `.fit()` is called.  Refitting overwrites previous state but includes contribution from all added observations.
- The `.add_observations()` and `.fit()` methods can be chained together if one-line training is desired: `detector.add_observations(observations).fit()`.
- An observation containing a feature name or feature value that has not been previously fit will be scored as `nan`.  To instead ignore any such "new" features and score an observation based on known features only, initialize the detector with `ignore_unknown=True`.

### Tests
To run unit tests:
```
$ python -m unittest discover -v
```
