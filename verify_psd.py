### ---------------------- IMPORTS ---------------------- ###
import numpy as np
import matplotlib.pyplot as plt
### ----------------------------------------------------- ###

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
        self.wait_time = 0.2 # in seconds

        # pass object attributes to class
        self.power_df = power_df                      
        self.index_df = index_df
        
        # set accepted to -1
        self.index_df['accepted'] = -1
        
        # get background
        self.index_df['facearray'] = 'w'
        
        # create figure and axis
        self.fig, self.axs = plt.subplots(2, 1, sharex = False, figsize=(8,8))

        # remove top and right axis
        self.axs[0].spines["top"].set_visible(False)
        self.axs[0].spines["right"].set_visible(False)
        self.axs[1].spines["top"].set_visible(False)
        self.axs[1].spines["right"].set_visible(False)
            
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
        
        # save verified index and power_df file
        self.index_df.to_csv(self.index_verified_path, index = False)
        self.power_df.to_pickle(self.power_mat_verified_path)

        print(f"Verified PSDs were saved in '{self.search_path}'.\n")    
        
        
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
        plot_data(self)
        plot self y and t and mark seizure
        """
        
        # get index
        self.get_index()

        # clear graph
        self.axs[0].clear()
        self.axs[1].clear() 
        
        # get PSD and SEM
        psd = np.mean(self.power_df['pmat'][self.i], axis = 1)
        sem = np.std(self.power_df['pmat'][self.i], axis = 1) / np.sqrt(self.power_df['pmat'][self.i].shape[1])
        time_plot = np.mean(self.power_df['pmat'][self.i], axis = 0)
        
        # plot time
        t = np.arange(0,time_plot.shape[0],1)
        self.axs[0].plot(t ,time_plot, color='black', linewidth=1.5, alpha=0.9, label = self.index_df.index[self.i])
        # from outlier_detection import get_outliers
        # outliers = get_outliers(time_plot, 60, 7)
        
        # show outliers
        outliers = self.power_df['outliers'][self.i]
        self.axs[0].plot(t[outliers], time_plot[outliers], color='orange', linestyle='', marker='x')
        self.axs[0].set_facecolor(self.index_df['facearray'][self.i]);
        self.axs[0].legend(loc = 'upper right')
        self.axs[0].set_xlabel('Time Bin')
        self.axs[0].set_ylabel('Mean Power')
        
        # plot PSD
        self.axs[1].plot(self.power_df['freq'][self.i], psd, color='black', linewidth=1.5, alpha=0.9, label = self.index_df.index[self.i])
        self.axs[1].fill_between(self.power_df['freq'][self.i], psd+sem, psd-sem, color = 'gray')
        self.axs[1].set_facecolor(self.index_df['facearray'][self.i]);
        self.axs[1].set_xlabel('Frequency (Hz)')
        self.axs[1].set_ylabel('Power (V^2/Hz)')

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
            
            
# if __name__ == '__main__':
#     import yaml,os
#     import pandas as pd
#     from load_index import load_index
#     from matplotlib.widgets import Button, SpanSelector, TextBox
#         # define path and conditions for filtering
#     filename = 'file_index.csv'
#     parent_folder = r'C:\Users\panton01\Desktop\pydsp_analysis'
#     path =  os.path.join(parent_folder, filename)
    
#     # enter filter conditions
#     filter_conditions = {'brain_region':['bla', 'pfc'], 'treatment':['baseline','vehicle']} #
    
#     # define frequencies of interest
#     with open('settings.yaml', 'r') as file:
#         settings = yaml.load(file, Loader=yaml.FullLoader)
    
#     #### ---------------------------------------------------------------- ####
    
#     # load index and power dataframe
#     index_df = load_index(path)
#     power_df = pd.read_pickle(r'C:\Users\panton01\Desktop\pydsp_analysis\power_mat.pickle')
       
#     # init gui object
#     callback = matplotGui(settings, index_df, power_df)
#     plt.subplots_adjust(bottom=0.15) # create space for buttons
    
#     # add title and labels
#     callback.fig.suptitle('Select PSDs', fontsize=12)        # title
#     # callback.fig.text(0.5, 0.09,'Frequency (Hz)', ha="center")                                          # xlabel
#     # callback.fig.text(.02, .5, 'Power (V^2/Hz)', ha='center', va='center', rotation='vertical')         # ylabel
#     callback.fig.text(0.9, 0.04,'**** KEY: Previous = <-, Next = ->, Accept = y, Reject = n, Accept all = a, Reject all = r ****' ,      # move/accept labels
#                       ha="right", bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9),))              
                                                    
#     # add key press
#     idx_out = callback.fig.canvas.mpl_connect('key_press_event', callback.keypress)
    
#     # set useblit True on gtkagg for enhanced performance
#     # span = SpanSelector(callback.axs, callback.keypress, 'horizontal', useblit=True,
#     #     rectprops=dict(alpha=0.5, facecolor='red'))
#     # plt.show()

































