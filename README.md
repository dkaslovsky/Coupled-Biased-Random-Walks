# Coupled-Biased-Random-Walks
Outlier detection for categorical data

### Overview
Lightweight Python 2/3 compatible implementation of the outlier detection algorithm described by Pang, Cao, and Chen in https://www.ijcai.org/Proceedings/16/Papers/272.pdf.

This implementation operates on Python dicts rather than Pandas DataFrames.  This has the advantage of allowing the model to be updated with new observations in a trivial manner and is more efficient in certain aspects.  However, these advantages come at the cost of iterating a (potentially large) dict of observed values more times than might otherwise be necessary using an underlying DataFrame implementation.

If one is working with data previously loaded into a DataFrame, simply use the result of `pandas.DataFrame.to_dict(orient='records')` instead of the DataFrame itself to add observations to the model.

### Tests
To run unit tests:
```
$ python -m unittest discover -v tests
```
