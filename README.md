# Coupled-Biased-Random-Walks
Outlier detection for categorical data

### Overview
Lightweight Python 2/3 compatible implementation of the Coupled Biased Random Walks (CBRW) outlier detection algorithm described by Pang, Cao, and Chen in https://www.ijcai.org/Proceedings/16/Papers/272.pdf.

This implementation operates on Python dicts rather than Pandas DataFrames.  This has the advantage of allowing the model to be updated with new observations in a trivial manner and is more efficient in certain aspects.  However, these advantages come at the cost of iterating a (potentially large) dict of observed values more times than might otherwise be necessary using an underlying DataFrame implementation.

If one is working with data previously loaded into a DataFrame, simply use the result of `pandas.DataFrame.to_dict(orient='records')` instead of the DataFrame itself to add observations to the model.


### Example
Let's run the CBRW detection algorithm on the authors' example data set from the paper:

<img src="./example_table.png" width="400">

This data is saved as a [.CSV file](./data/) in this repository and is loaded into memory as a list of dicts by [example.py](./example.py).

Note that we drop the "Cheat?" column as this is essentially the target variable indicating the anomalous activity to be detected.  The detector is instantiated as observations are added as follows:
```
cbrw = CBRW()
cbrw.add_observations(observations)
```
where `observations` is an iterable of dicts such as the one loaded from the example .CSV file.

### Tests
To run unit tests:
```
$ python -m unittest discover -v
```
