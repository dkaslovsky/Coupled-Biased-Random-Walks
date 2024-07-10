## 2.1.1 / 2024-07-10
* [Changed] updated dependencies to latest versions

## 2.1.0 / 2020-09-05
* [Added] version is exposed as `coupled_biased_random_walks.__version__`

## 2.0.0 / 2020-08-30
* [Added] type hints
* [Changed] removed support for Python 2 and <3.7
* [Changed] updated dependencies to latest versions

## 1.1.1 / 2020-08-29
* [Fixed] enforce feature weight and stationary probability sum normalization

## 1.1.0 / 2020-08-23
* [Added] `CBRW.value_scores()` function to return individual value scores of an observation
* [Added] exceptions inherit from base `CBRWError` exception
* [Added] test cases for NaN values
* [Changed] README describes value_scores functionality
* [Changed] example.py is reorganized to include value_scores
* [Fixed] tests compare sorted lists
