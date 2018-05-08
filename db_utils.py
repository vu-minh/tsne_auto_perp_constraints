""" Pandas HDFStore utils

    Goal:
        + Avoiding inconsistencies when writing to a store
        from multiple processes/threads
        + Different processes/threads working on a same store will simply queue

    Link:
        https://stackoverflow.com/questions/22522551/pandas-hdf5-as-a-database/29014295#29014295

    Usage:
    ```
        result = do_long_operations()
        with SafeHDFStore('example.hdf') as store:
            # Only put inside this block the code which operates on the store
            store['result'] = result
    ```
"""

import pandas as pd
from pandas import HDFStore
import os
import time

IS_PARALLEL = True


class SafeHDFStore(HDFStore):
    def __init__(self, *args, **kwargs):
        if IS_PARALLEL:
            if len(args) < 1 or len(args[0]) == 0:
                raise ValueError("The first parameter should be database name")

            self._db_name = args[0]
            probe_interval = kwargs.pop("probe_interval", 0.5)
            self._lock = "%s.lock" % args[0]
            while True:
                try:
                    # print('[{}] Waiting for lock ...'.format(self._db_name))
                    self._flock = os.open(self._lock,
                                          os.O_CREAT |
                                          os.O_EXCL |
                                          os.O_WRONLY)
                    break
                except FileExistsError:
                    time.sleep(probe_interval)

        HDFStore.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        HDFStore.__exit__(self, *args, **kwargs)

        if IS_PARALLEL:
            os.close(self._flock)
            os.remove(self._lock)


DB_PATH = './db_files'
DB_EXT = 'h5'


def show_db(db_name, key=None):
    db_name = '{}/{}.h5'.format(DB_PATH, db_name)
    with SafeHDFStore(db_name) as store:
        print(store)

        if key is not None:
            print(store[key])


def load_data(db_name, key, columns=[]):
    db_name = '{}/{}.h5'.format(DB_PATH, db_name)
    with SafeHDFStore(db_name) as store:
        if columns:
            param = "columns={}".format(str(columns))
            df = store.select(key, param)
        else:
            df = store.select(key)
    return df


def append_to_db(db_name, key, records):
    """ Append a new record to existed db
        Create auto-increasing id for each record
    """
    db_name = '{}/{}.{}'.format(DB_PATH, db_name, DB_EXT)
    with SafeHDFStore(db_name) as store:
        try:
            n_rows_existed = store.get_storer(key).nrows
        except Exception:  # empty database
            n_rows_existed = 0

        n_rows_to_add = len(records)
        new_indices = range(n_rows_existed, n_rows_existed + n_rows_to_add)

        # create temp dataframe to hold the record
        df = pd.DataFrame(records, index=new_indices)

        # append new dataframe into database
        store.append(key, df)
        # print('[{}]Append to DB with key={} succesfully!'.format(db_name, key))


def reset_key(db_name, key):
    """ Clear all data of a specified key
    """
    db_name = '{}/{}.{}'.format(DB_PATH, db_name, DB_EXT)
    with SafeHDFStore(db_name) as store:
        del store[key]


if __name__ == '__main__':
    db_name = 'hello'
    show_db(db_name)

    print("Test add record")
    a_record = dict(key1=99, key2=100, key3=0.4256)

    append_to_db(db_name, key='test', records=[a_record])
    show_db(db_name, 'test')

    print("Test add multiple records")
    records = [a_record] * 2
    append_to_db(db_name, key='test', records=records)
    show_db(db_name, 'test')

    # reset_key(db_name, key='test')
    # show_db(db_name)

    # load data from database to pandas dataframe
    df = load_data(db_name, key='test', columns=['key3', 'key2'])
    print(type(df))
    print(df)
