# -*- coding: utf-8 -*-

########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
import pandas as pd

# type checking
from beartype import beartype
from typing import TypeVar
PandasDf = TypeVar('pandas.core.frame.DataFrame')
########## ---------------------------------------------------------------- ##########


@beartype
def load_index(path:str) -> PandasDf:
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


@beartype
def filter_index(index_df:PandasDf, filter_conditions:dict) -> PandasDf:
    """
    Filter index dataframe

    Parameters
    ----------
    index_df : PandasDf, dataframe containing experiment index
    filter_conditions : dict, with keys = columns and values = lists of strings to filter


    Returns
    -------
    PandasDf, filtered experiment index

    """

    for category, groups in filter_conditions.items():
        
        # create empty index
        idx = np.zeros(len(index_df), dtype = bool)
        
        for value in groups: # iterate over list of strings to filter
            
            # or statement between groups of same category
            idx = (idx) | (index_df[category] == value)
            
        # filter signal
        index_df = index_df[idx]

    return index_df

@beartype
def load_n_filter(path:str, filter_conditions:dict) -> PandasDf:
    """
    Load and filter index array

    Parameters
    ----------
    path : str, path to load index
    filter_conditions : dict, with keys = columns and values = lists of strings to filter

    Returns
    -------
    PandasDf, filtered experiment index

    """
    
    # load index csv as dataframe
    index_df = load_index(path)
    
    # filter index based on conditions
    index_df = filter_index(index_df, filter_conditions).reset_index(drop=True)
    
    return index_df
    
    

if __name__ == '__main__':
    
    # define path and conditions for filtering
    path = r'C:\Users\panton01\Desktop\pydsp_analysis\index.csv'
    filter_conditions = {'brain_region':['bla'], 'treatment':['baseline','vehicle']}
    
    # load df
    index_df = load_index(path)
    
    # filter index based on conditions
    index_df_filt = load_n_filter(path, filter_conditions)
    

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    