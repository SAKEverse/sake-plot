# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:06:07 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import os
import adi
import numpy as np
from beartype import beartype
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
    
    # input parameter names required
    input_parameters = ['folder_path', 'file_name', 'channel_id', 'block', 'start_time', 'stop_time']
    
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
    def get_data_adi(self) -> np.ndarray: 
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
