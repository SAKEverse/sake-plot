# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:37:05 2020
@author: panton01
"""

### ---------------------- IMPORTS ---------------------- ###
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
### ----------------------------------------------------- ###


class matplotGui(object):
    """
        Matplotlib GUI for user seizure verification.
    """
    
    ind = 0 # set internal counter
       
    def __init__(self, save_path, index_df, power_df):

        
        # pass object attributes to class
        self.save_path = save_path
        self.power_df = power_df                      
        self.index_df = index_df
        
        # set accepted to -1
        self.index_df['accepted'] = -1
        
        # get background
        self.index_df['facearray'] = 'w'
        
        # create figure and axis
        self.fig, self.axs = plt.subplots(1, 1, sharex = False, figsize=(8,8))

        # remove top and right axis
        self.axs.spines["top"].set_visible(False)
        self.axs.spines["right"].set_visible(False)
            
        # create first plot
        self.plot_data()
    
    def save_idx(self):
        """
        Saves accepted PSD index and mat files
        Returns
        -------
        None.
        """

        # check if all PSDs were verified
        if np.any(self.index_df['accepted'] == -1) == True:
            print('Some PSDs were not verified.')
            
        # get accepted PSDs
        accepted_idx = self.index_df['accepted'] == 1
        self.index_df = self.index_df[accepted_idx]
        self.power_df = self.power_df[accepted_idx]
        
        # drop extra columns
        self.index_df = self.index_df.drop(columns = ['accepted','facearray'])
        
        # save curated index file
        self.index_df.to_csv('file_index_accepted.csv')
        self.power_df.to_pickle('power_mat_accepted.pickle')
        # save curated power mat file

        print('PSDs were saved.\n')    
        
        
    def get_index(self):
        """

        Returns
        -------
        None.
        """
        
        # reset counter when limits are exceeded
        if self.ind >= len(index_df):
            self.ind = 0 
            
        elif self.ind < 0:
            self.ind = len(index_df)-1
       
        # set counter to internal counter
        self.i = self.ind

        
        
    def plot_data(self, **kwargs):
        """
        plot_data(self)
        plot self y and t and mark seizure
        """
        # get index
        self.get_index()

        # clear graph
        self.axs.clear() 
        
        # get PSD and SEM
        psd = np.mean(self.power_df['pmat'][self.i], axis = 1)
        sem = np.std(self.power_df['pmat'][self.i], axis = 1) / np.sqrt(self.power_df['pmat'][self.i].shape[1])
        
        # plot new graph
        self.axs.plot(self.power_df['freq'][self.i], psd, color='black', linewidth=1.5, alpha=0.9, label = self.index_df.index[self.i])
        self.axs.fill_between(self.power_df['freq'][self.i], psd+sem, psd-sem, color = 'gray')
        self.axs.set_facecolor(self.index_df['facearray'][self.i]);
        self.axs.legend(loc = 'upper right')
        
        self.fig.canvas.draw() # draw

         
    ## ------  Keyboard press ------ ##     
    def keypress(self,event):
        if event.key == 'right':
            self.ind += 1 # add one to class index
            self.plot_data() # plot
        if event.key == 'left':
            self.ind -= 1 # subtract one to class index
            self.plot_data() # plot
        if event.key == 'y':
           self.index_df.at[self.i, 'facearray'] = 'palegreen'
           self.axs.set_facecolor('palegreen')
           self.index_df.at[self.i, 'accepted'] = 1
           self.fig.canvas.draw()
        if event.key == 'n':
            self.index_df.at[self.i, 'facearray'] = 'salmon'
            self.axs.set_facecolor('salmon')
            self.index_df.at[self.i, 'accepted'] = 0
            self.fig.canvas.draw()
        if event.key == 'enter':
            plt.close()
            self.save_idx() # save file to csv
            
            
    #         ## ------ Mouse Button Press ------ ##   
    # def forward(self, event):
    #     self.ind += 1 # add one to class index
    #     self.plot_data() # plot
        
    # def previous(self, event):
    #     self.ind -= 1 # subtract one to class index
    #     self.plot_data() # plot
        
    # def accept(self, event):
    #     self.facearray[self.i] = 'palegreen'
    #     self.axs.set_facecolor('palegreen')
    #     self.index_df['accepted'][self.i] = 1
    #     self.fig.canvas.draw()
        
    # def reject(self, event):
    #     self.facearray[self.i] = 'salmon'
    #     self.axs.set_facecolor('salmon')
    #     self.index_df['accepted'][self.i] = 1
    #     self.fig.canvas.draw()


if __name__ == '__main__':
    import yaml
    from load_index import load_index
    from matplotlib.widgets import Button, SpanSelector, TextBox
        # define path and conditions for filtering
    filename = 'file_index.csv'
    parent_folder = r'C:\Users\panton01\Desktop\pydsp_analysis'
    path =  os.path.join(parent_folder, filename)
    
    # enter filter conditions
    filter_conditions = {'brain_region':['bla', 'pfc'], 'treatment':['baseline','vehicle']} #
    
    # define frequencies of interest
    with open('settings.yaml', 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)
    
    #### ---------------------------------------------------------------- ####
    
    # load index and power dataframe
    index_df = load_index(path)
    power_df = pd.read_pickle(r'C:\Users\panton01\Desktop\pydsp_analysis\power_mat.pickle')
       
    # init gui object
    callback = matplotGui(parent_folder, index_df, power_df)
    plt.subplots_adjust(bottom=0.15) # create space for buttons
    
    # add title and labels
    callback.fig.suptitle('Select PSDs', fontsize=12)        # title
    callback.fig.text(0.5, 0.09,'Frequency (Hz)', ha="center")                                          # xlabel
    callback.fig.text(.02, .5, 'Power (V^2/Hz)', ha='center', va='center', rotation='vertical')         # ylabel
    callback.fig.text(0.9, 0.04,'**** KEY ** Previous : <-, Next: ->, Accept: Y, Reject: N ****' ,      # move/accept labels
                      ha="right", bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9),))              
                                                    
    # add key press
    idx_out = callback.fig.canvas.mpl_connect('key_press_event', callback.keypress)
    
    # set useblit True on gtkagg for enhanced performance
    span = SpanSelector(callback.axs, callback.keypress, 'horizontal', useblit=True,
        rectprops=dict(alpha=0.5, facecolor='red'))
    plt.show()

































