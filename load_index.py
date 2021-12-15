# -*- coding: utf-8 -*-

import pandas as pd

def load_index(path:str):
    """
    Load experiment index from csv file

    Parameters
    ----------
    path : str, path to load index

    Returns
    -------
    PandasDf, with experiment index

    """
    return pd.read_csv(path)