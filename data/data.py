from csv import DictReader
from functools import partial

from six import iteritems


def load_from_csv(path_to_csv, exclude_cols=None):
    with open(path_to_csv, 'r') as csvfile:
        data = list(DictReader(csvfile))
    if exclude_cols is not None:
        if isinstance(exclude_cols, str):
            exclude_cols = {exclude_cols}
        filt = partial(filter_fields, fields=set(exclude_cols))
        return [filt(rec) for rec in data]
    return data


def filter_fields(record, fields):
    return {k: v for k, v in iteritems(record) if k not in fields}
