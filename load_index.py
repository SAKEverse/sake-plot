# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:37:47 2021

@author: panton01
"""

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