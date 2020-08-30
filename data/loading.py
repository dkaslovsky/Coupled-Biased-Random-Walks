from csv import DictReader
from functools import partial
from typing import Iterable, List, Optional, Set

from coupled_biased_random_walks.types import observation_type


def load_from_csv(
    path_to_csv: str,
    exclude_cols: Optional[Iterable[str]] = None,
) -> List[observation_type]:
    """
    Load a CSV and return a list of dicts, one dict for each row of
    the form {column_header1: <value>, column_header2: <value>, ...}
    :param path_to_csv: path to CSV file
    :param exclude_cols: iterable of columns to exclude (often the target variable)
    """
    with open(path_to_csv, 'r') as csvfile:
        # use list to load into memory before closing
        data = list(DictReader(csvfile))
    if exclude_cols is None:
        return data
    # filter based on exclude cols
    if isinstance(exclude_cols, str):
        exclude_cols = {exclude_cols}
    filt = partial(filter_keys, fields=set(exclude_cols))
    return [filt(rec) for rec in data]


def filter_keys(record: observation_type, fields: Set[str]):
    """
    Filter keys from a dict
    :param record: dict
    :param fields: set of strings indicating fields to drop
    :return:
    """
    return {k: v for k, v in record.items() if k not in fields}
