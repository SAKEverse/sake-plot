### ---------------------- IMPORTS ---------------------- ###
import numpy as np
import matplotlib.pyplot as plt
from stft import f_fill
from outlier_detection import get_outliers
from tqdm import tqdm
### ----------------------------------------------------- ###


def remove_outliers(pmat:np.ndarray, outlier_window, outlier_threshold):
    """
    Remove outliers based on MAD

    Parameters
    ----------
    pmat : np.ndarray

    Returns
    -------
    pmat : np.ndarray
    outliers : np.ndarray

    """
    # get outliers
    outliers = get_outliers(np.mean(pmat, axis=0), outlier_window, outlier_threshold)

    # replace outliers with nans
    pmat[:, outliers] = np.nan
    
    # interpolate missing data
    pmat = f_fill(pmat, axis=1)
    
    # fill with median value
    row_med = np.nanmedian(pmat, axis=1)

    # find indices that you need to replace
    inds = np.where(np.isnan(pmat))
    
    # place row medians in the indices.
    pmat[inds] = np.take(row_med, inds[0])
    
    return pmat, outliers

class matplotGui:
    """
        Matplotlib GUI for user seizure verification.
    """
    
    ind = 0 # set internal counter
       
    def __init__(self, settings, index_df, power_df):

        # get values from dictionary
        for key, value in settings.items():
               setattr(self, key, value)
               
        # wait time after plot
        self.wait_time = 0.1 # in seconds

        # pass object attributes to class
        self.power_df = power_df                      
        self.index_df = index_df
        
        # set accepted to -1
        self.index_df['accepted'] = -1
        
        # get background
        self.index_df['facearray'] = 'w'
        
        # # create figure and axis
        self.fig, self.axs = plt.subplots(2, 1, sharex = False, figsize=(8,8))

        # remove top and right axis
        self.axs[0].spines["top"].set_visible(False)
        self.axs[0].spines["right"].set_visible(False)
        self.axs[1].spines["top"].set_visible(False)
        self.axs[1].spines["right"].set_visible(False)
            
        # create first plot
        self.plot_data()
        plt.show()
        
    def get_index(self):
        """

        Returns
        -------
        None.
        """

        # reset counter when limits are exceeded
        if self.ind >= len(self.index_df):
            self.ind = 0 
            
        elif self.ind < 0:
            self.ind = len(self.index_df)-1
       
        # set counter to internal counter
        self.i = self.ind

        
        
    def plot_data(self, **kwargs):
        """
        Plot power area time plot and PSD for outlier and experiment verification
        """
        
        # get index
        self.get_index()

        # clear graph
        self.axs[0].clear()
        self.axs[1].clear() 
        
        # get pmat and freq
        pmat = self.power_df['pmat'][self.i].copy()
        freq = self.power_df['freq'][self.i].copy()
        
        # get time plot
        time_plot = np.mean(pmat, axis = 0)
        t = np.arange(0, time_plot.shape[0], 1) * self.fft_overlap * self.fft_win
        
        # plot time plot (before outlier removal)
        self.axs[0].plot(t ,time_plot, color='black', label = self.index_df.index[self.i])
        
        # plot PSD (before outlier removal)
        psd = np.mean(pmat, axis = 1)
        sem = np.std(pmat, axis = 1) / np.sqrt(pmat.shape[1])
        self.axs[1].plot(freq, psd, color='black', linewidth=1.5, alpha=0.9, label = self.index_df.index[self.i])
        self.axs[1].fill_between(freq, psd+sem, psd-sem, color = 'black', alpha=0.2)
        
        # remove outliers
        pmat, outliers = remove_outliers(pmat, self.outlier_window, self.outlier_threshold)
        
        # plot time plot (after outlier removal)
        self.axs[0].plot(t[outliers], time_plot[outliers], color='orange', linestyle='', marker='x')
        
        # plot PSD (after outlier removal)
        psd = np.mean(pmat, axis = 1)
        sem = np.std(pmat, axis = 1) / np.sqrt(pmat.shape[1])
        self.axs[1].plot(freq, psd, color='orange', linewidth=1.5, alpha=0.9, label = self.index_df.index[self.i])
        self.axs[1].fill_between(freq, psd+sem, psd-sem, color = 'orange', alpha=0.2)
        self.axs[1].set_ylim(np.min(psd), np.max(psd))
        
        # add labels and background colors
        self.axs[0].set_facecolor(self.index_df['facearray'][self.i]);
        self.axs[0].legend(loc = 'upper right')
        self.axs[0].set_xlabel('Time (Seconds)')
        self.axs[0].set_ylabel('Mean Power')
        self.axs[1].set_facecolor(self.index_df['facearray'][self.i]);
        self.axs[1].legend(['Outlier_threshold = {:.1f}'.format(self.outlier_threshold)], loc = 'upper right')
        self.axs[1].set_xlabel('Frequency (Hz)')
        self.axs[1].set_ylabel('Power (V^2/Hz)')
        plt.subplots_adjust(bottom=0.15)
        self.fig.suptitle('Select PSDs', fontsize=12)   
        self.fig.text(0.9, 0.04, '**** KEY: Previous = <-, Next = ->, Accept = y, Reject = n, Accept all = a, Reject all = r ****' ,      # move/accept labels
                      ha="right", bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9),))
        

        self.fig.canvas.callbacks.connect('key_press_event', self.keypress)
        plt.draw()
        
        

            
    def save_idx(self):
        """
        Saves accepted PSD index and mat files
        Returns
        -------
        None.
        """

        # check if all PSDs were verified
        if np.any(self.index_df['accepted'] == -1) == True:
            print('\n****** Some PSDs were not verified ******\n')
            
        # get accepted PSDs
        accepted_idx = self.index_df['accepted'] == 1
        self.index_df = self.index_df[accepted_idx]
        self.power_df = self.power_df[accepted_idx]
        
        # drop extra columns
        self.index_df = self.index_df.drop(columns = ['accepted','facearray'])
        
        # reset index
        self.index_df.reset_index(drop = True, inplace = True)
        self.power_df.reset_index(drop = True, inplace = True)
        
        # remove outliers
        for i in tqdm(range(len(self.power_df))):
            pmat, outliers = remove_outliers(self.power_df['pmat'][i], self.outlier_window, self.outlier_threshold)
        
        # save verified index and power_df file
        self.index_df.to_csv(self.index_verified_path, index = False)
        self.power_df.to_pickle(self.power_mat_verified_path)

        print(f"Verified PSDs were saved in '{self.search_path}'.\n")  
        
         
    ## ------  Keyboard press ------ ##     
    def keypress(self, event):
        # print(event.key)
        if event.key == 'right':
            self.ind += 1 # add one to class index
            self.plot_data() # plot
            
        if event.key == 'left':
            self.ind -= 1 # subtract one to class index
            self.plot_data() # plot
        
        if event.key == 'up':
            self.outlier_threshold += 0.5
            self.plot_data() # plot
            
        if event.key == 'down':
            self.outlier_threshold -= 0.5
            self.plot_data() # plot

        if event.key == 'y':
           # set values to arrays
           self.index_df.at[self.i, 'facearray'] = 'palegreen'
           self.index_df.at[self.i, 'accepted'] = 1
           
           # change background color  
           self.axs[0].set_facecolor('palegreen')
           self.axs[1].set_facecolor('palegreen')
           
           # draw and pause for user visualization
           plt.draw()
           plt.pause(self.wait_time)
           
           # plot next event
           self.ind += 1 # add one to class index
           self.plot_data() # plot
          
        if event.key == 'n':
            # set values to arrays
            self.index_df.at[self.i, 'facearray'] = 'salmon'
            self.index_df.at[self.i, 'accepted'] = 0
            
            # change background color
            self.axs[0].set_facecolor('salmon')
            self.axs[1].set_facecolor('salmon')
            
            # draw and pause for user visualization
            plt.draw()
            plt.pause(self.wait_time)
            
            # plot next event
            self.ind += 1 # add one to class index
            self.plot_data() # plot
            
        if event.key == 'a':
           # set values to arrays
           self.index_df.at[:, 'facearray'] = 'palegreen'
           self.index_df.at[:, 'accepted'] = 1
           
           # change background color  
           self.axs[0].set_facecolor('palegreen')
           self.axs[1].set_facecolor('palegreen')
           
           # draw and pause for user visualization
           plt.draw()
           plt.pause(self.wait_time)

        if event.key == 'r':
            # set values to arrays
            self.index_df.at[:, 'facearray'] = 'salmon'
            self.index_df.at[:, 'accepted'] = 0
            
            # change background color  
            self.axs[0].set_facecolor('salmon')
            self.axs[1].set_facecolor('salmon')
            
            # draw and pause for user visualization
            plt.draw()
            plt.pause(self.wait_time)
        
        if event.key == 'enter':
            plt.close()
            self.save_idx() # save file to csv
            
            
if __name__ == '__main__':
    import yaml,os
    import pandas as pd
    from load_index import load_index
        # define path and conditions for filtering
    filename = 'index.csv'
    parent_folder = r'C:\Users\panton01\Desktop\pydsp_analysis'
    path =  os.path.join(parent_folder, filename)
    
    # enter filter conditions
    filter_conditions = {'brain_region':['bla', 'pfc'], 'treatment':['baseline','vehicle']} #
    
    # define frequencies of interest
    with open('settings.yaml', 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)
        
    settings.update({'outlier_threshold':5, 'outlier_window':11})
    
    #### ---------------------------------------------------------------- ####
    
    # load index and power dataframe
    index_df = load_index(path)
    power_df = pd.read_pickle(r'C:\Users\panton01\Desktop\pydsp_analysis\power_mat.pickle')
       
    
    pmat, outliers = remove_outliers(power_df['pmat'][0], 11, 5)
    # # init gui object
    # callback = matplotGui(settings, index_df, power_df)
    # plt.subplots_adjust(bottom=0.15) # create space for buttons
    
    # # add title and labels
    # callback.fig.suptitle('Select PSDs', fontsize=12)        # title
    # callback.fig.text(0.9, 0.04,'**** KEY: Previous = <-, Next = ->, Accept = y, Reject = n, Accept all = a, Reject all = r ****' ,      # move/accept labels
    #                   ha="right", bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9),))              
                                                    
    # # add key press
    # idx_out = callback.fig.canvas.mpl_connect('key_press_event', callback.keypress)































