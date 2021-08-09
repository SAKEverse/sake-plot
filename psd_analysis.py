# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:56:55 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import os
import adi
import numpy as np
import pandas as pd
from beartype import beartype
from typing import TypeVar
PandasDf = TypeVar('pandas.core.frame.DataFrame')
########## ---------------------------------------------------------------- ##########

class AdiGet:
    """
    Class to get data from labchart.
    
    
    Requires a dictionary with following key/value pairs:
    
    -----------------------------------------------------
    folder_path: str, parent folder
    file_name: str, file name
    channel_id:, int: channel id
    block, int: block number
    start_time, int: start time in samples for file read
    stop_time, int:  stop time in samples for file read
    -----------------------------------------------------
     
    """
    
    @beartype
    def __init__(self, propeties:dict):
        """
        
        Parameters
        ----------
        propeties : dict

        Returns
        -------
        None.

        """
        
        # get values from dictionary
        for key, value in propeties.items():
               setattr(self, key, value)
       
        # get load path
        self.file_path = os.path.join(self.folder_path, self.file_name)


    @beartype
    def get_data_adi(self) -> np.ndarray : 
        """
        Get data from labchart channel object

        Returns
        -------
        np.ndarray: 1D array

        """
        
        # get adi read object
        fread = adi.read_file(self.file_path)
        
        # get channel object
        ch_obj = fread.channels[self.channel_id]

        # get data 
        data = ch_obj.get_data(self.block+1, start_sample=self.start_time, stop_sample=self.stop_time)
    
        del fread # delete fread object
       
        return data
        
        
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
    

if __name__ == '__main__':
    
    path = r'C:\Users\panton01\Desktop\index.csv'
    filter_conditions = {'brain_region':['bla'], 'treatment':['baseline','vehicle']}
    
    # properties needed to 
    adi_properties = ['folder_path', 'file_name', 'channel_id', 'block', 'start_time', 'stop_time']
    
    # get index
    index_df = load_index(path)
    
    # filter index based on conditions
    index_df_filt = filter_index(index_df, filter_conditions).reset_index(drop=True)
    
    # get data
    properties = index_df_filt[adi_properties].loc[0].to_dict()
    signal = AdiGet(properties).get_data_adi()
    
    # run PSD

    
    
    
    
    
    
    
    
    