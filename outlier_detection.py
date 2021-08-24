# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:09:35 2021

@author: gweiss01
"""
import numpy as np

def fast_outliers(arr,window,threshold):
    #first make a threshold index:
    #divide the array into windows by reshaping, then find the median of each window, this makes an array of size: original_len/window
    medians=np.median(arr[:(len(arr)//window)*window].reshape(-1,window),axis=1)
    #expand the array back to full size by interpolaing the in between values
    smoothed_medians=np.interp(np.linspace(0,len(medians),len(medians)*window),np.arange(len(medians)),medians)
    #fill the end values 
    time_thresholds=np.append(smoothed_medians,[smoothed_medians[-1]]*(len(arr)-len(smoothed_medians)))
    #compare the original array to the threshold index to find the outliers
    outliers = ((arr>(time_thresholds*threshold)) |  (arr<-(time_thresholds*threshold)))
    return outliers