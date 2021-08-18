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
       
    def __init__(self, index_df, power_df, n_plot = [4, 1]):
        """   
        Parameters
        ----------
        data : 3D Numpy array, (1D = seizure segments, 2D =  columns (samples: window*sampling rate), 3D = channels ) 
        idx_bounds : 2D Numpy array (1D = seizure segments, 2D, 1 = start, 2 = stop index)
        obj: UserVerify object
        file_id: Str, file anme
        """
        
        # pass object attributes to class
        self.power_df = power_df                      
        self.index_df = index_df 
        self.nplots = n_plot[0] * n_plot[1]
        # self.facearray = ['w']*idx_bounds.shape[0]              # color list
        
        # create figure and axis
        self.fig, self.axs = plt.subplots(n_plot[0], n_plot[1], sharex = False, figsize=(8,8))

        # remove all axes except left 
        for i in range(self.axs.shape[0]): 
            self.axs[i].spines["top"].set_visible(False)
            self.axs[i].spines["right"].set_visible(False)
            
        # create first plot
        self.plot_data()
    
    def save_idx(self):
        """
        Save user predictions to csv file as binary
        Returns
        -------
        None.
        """
        # pre allocate file with zeros
        ver_pred = np.zeros(self.data.shape[0])


        for i in range(self.idx_out.shape[0]): # assign index to 1
        
            if self.idx_out[i,0] > 0:
                # add 1 to stop bound because of python indexing
                ver_pred[self.idx_out[i,0]:self.idx_out[i,1]+1] = 1
            
        # save file
        np.savetxt(os.path.join(self.verpred_path,self.file_id), ver_pred, delimiter=',',fmt='%i')
        print('Verified predictions for ', self.file_id, ' were saved.\n')    
        
        
    def get_index(self):
        """
        get cntr
        Returns
        -------
        None.
        """

        self.cnt = self.ind

        
        
    def plot_data(self, **kwargs):
        """
        plot_data(self)
        plot self y and t and mark seizure
        """
        # get index, start and stop times
        self.get_index()
        
    
        # ###  PLot first channel first with different settings  ###   
        # # get seizure with nearby segments
        # i = 0  # first channel
        # y = self.data[self.start - self.seg : self.stop + self.seg,:, i].flatten()
        # t = np.linspace(self.start - self.seg, self.stop + self.seg, len(y))# get time
        # self.axs[i].clear() # clear graph
        # self.axs[i].plot(t, y, color='k', linewidth=0.75, alpha=0.9, label= timestr) 
        # self.axs[i].set_facecolor(self.facearray[self.i]);
        # 
        # self.axs[i].set_title(self.ch_list[i], loc ='left')
        
                 
        # plot remaining channels    
        for plot_cntr,i in enumerate(range(self.cnt, self.cnt + self.nplots)): 
            # Plot seizure with surrounding region
            self.axs[plot_cntr].clear() # clear graph
            
            psd = np.mean(self.power_df['pmat'][i], axis = 1)
            self.axs[plot_cntr].plot(self.power_df['freq'][i], psd, color='gray', linewidth=0.75, alpha=0.9, label = self.index_df.index[i])
            self.axs[i].legend(loc = 'upper right')
        
        self.fig.canvas.draw() # draw


    ## ------ Mouse Button Press ------ ##   
    def forward(self, event):
        self.ind += 1 # add one to class index
        self.plot_data() # plot
        
    def previous(self, event):
        self.ind -= 1 # subtract one to class index
        self.plot_data() # plot
        
    def accept(self, event):
        self.facearray[self.i] = 'palegreen'
        self.axs[0].set_facecolor('palegreen')
        if self.idx_out[self.i,1] == -1:
            self.idx_out[self.i,:] = self.idx[self.i,:]
        else:
            self.idx_out[self.i,:] = self.idx_out[self.i,:]
        self.fig.canvas.draw()
        
    def reject(self, event):
        self.facearray[self.i] = 'salmon'
        self.axs[0].set_facecolor('salmon')
        self.idx_out[self.i,:] = -1
        self.fig.canvas.draw()
        
    def submit(self, text): # to move to a certain seizure number
        self.ind = eval(text)
        self.plot_data() # plot

             
    ## ------  Keyboard press ------ ##     
    def keypress(self,event):
        if event.key == 'right':
            self.ind += 1 # add one to class index
            self.plot_data() # plot
        if event.key == 'left':
            self.ind -= 1 # subtract one to class index
            self.plot_data() # plot
        if event.key == 'y':
            self.facearray[self.i] = 'palegreen'
            self.axs[0].set_facecolor('palegreen')
            if self.idx_out[self.i,1] == -1:
                self.idx_out[self.i,:] = self.idx[self.i,:]
            else:
                self.idx_out[self.i,:] = self.idx_out[self.i,:]
                self.fig.canvas.draw()
        if event.key == 'n':
            self.facearray[self.i] = 'salmon'
            self.axs[0].set_facecolor('salmon')
            self.idx_out[self.i,:] = -1  
            self.fig.canvas.draw()
        if event.key == 'enter': 
            plt.close()
            self.save_idx() # save file to csv


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
    callback = matplotGui(index_df, power_df)
    plt.subplots_adjust(bottom=0.15) # create space for buttons
    
    # add title and labels
    callback.fig.suptitle('Select PSDs', fontsize=12)        # title
    callback.fig.text(0.5, 0.09,'Frequency (Hz)', ha="center")                                          # xlabel
    callback.fig.text(.02, .5, 'Power (V^2/Hz)', ha='center', va='center', rotation='vertical')         # ylabel
    callback.fig.text(0.9, 0.04,'** KEY = Previous : <-, Next: ->, Accept: Y, Reject: N **' ,           # move/accept labels
                      ha="right", bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9),))              
                                                    
    # add key press
    idx_out = callback.fig.canvas.mpl_connect('key_press_event', callback.keypress)
    
    # set useblit True on gtkagg for enhanced performance
    span = SpanSelector(callback.axs[0], callback.keypress, 'horizontal', useblit=True,
        rectprops=dict(alpha=0.5, facecolor='red'))
    plt.show()

































