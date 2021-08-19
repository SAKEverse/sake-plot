#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:10:37 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import os, yaml, click
import pandas as pd
from filter_index import load_index
########## ---------------------------------------------------------------- ##########


@click.group()
@click.pass_context
def main(ctx):
    """

    ███████ ██    ██ ██████  ███████ ███████                                     
    ██   ██  ██  ██  ██   ██ ██      ██   ██                                      
    ███████   ████   ██   ██ ███████ ███████                                                                            
    ██         ██    ██   ██      ██ ██                                           
    ██         ██    ██████  ███████ ██                                           
             
     
    """
    
    # get settings and pass to context
    with open(settings_path, 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)
        ctx.obj = settings.copy()
        
    # save original settings
    ctx.obj.update({'settings': settings}) 
        
    # set files present to zero
    ctx.obj.update({'index_present': 0})
    ctx.obj.update({'power_present': 0})
    
    # get path to files
    ctx.obj.update({'index_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_index'])})
    ctx.obj.update({'power_mat_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_power_mat'])})
    ctx.obj.update({'index_verified_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_index_verified'])})
    ctx.obj.update({'power_mat_verified_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_power_mat_verified'])})
    
    # check if index file is present and get full path
    if os.path.isfile(ctx.obj['index_path']):
        ctx.obj.update({'index_present':1})
        
    # check if power mat file is present and get full path
    if os.path.isfile(ctx.obj['power_mat_path']):
        ctx.obj.update({'power_present':1})
    
            
### ------------------------------ SET PATH ------------------------------ ### 
@main.command()
@click.argument('path', type = str)
@click.pass_context
def setpath(ctx, path):
    """Set path to index file parent directory"""    
    # add path to original settings
    ctx.obj['settings']['search_path'] = path

    # write to file
    with open(settings_path, 'w') as file:
        yaml.dump(ctx.obj['settings'], file)
        
    click.secho(f"\n -> Path was set to:'{path}'.\n", fg = 'green', bold = True)


### ------------------------------ STFT ---------------------------------- ###     
@main.command()
@click.option('--freq', type = str, help = 'Enter frequency range, e.g. 1-30')
@click.pass_context
def stft(ctx, freq):
    """Runs the Short-time Fourier Transform"""
    from psd_analysis import get_pmat

    # check if index file was not found
    if not ctx.obj['index_present']:
        click.secho(f"\n -> File '{ctx.obj['file_index']}' was not found in '{ctx.obj['search_path']}'.\n", fg = 'yellow', bold = True)
        return
    
    # get frequency
    if freq is not None:
        freq_range = [int(i) for i in freq.split('-')]
        
        if len(freq_range) !=2:
              click.secho(f"\n -> '{freq}' could not be parsed. Please use the following format: 1-30.\n", fg = 'yellow', bold = True)
              return
    
    # load index 
    index_df = load_index(ctx.obj['index_path'])
                          
    # get power 
    power_df = get_pmat(index_df, fft_duration = ctx.obj['fft_win'],
                        freq_range = ctx.obj['fft_freq_range'], f_noise = ctx.obj['mains_noise'])
    
    # save index and power
    index_df.to_csv(ctx.obj['index_verified_path'], index = False)
    power_df.to_pickle(ctx.obj['power_mat_path'])
    power_df.to_pickle(ctx.obj['power_mat_verified_path'])
    
    # get freq_range for display
    freq_range = '-'.join(list(map(str, ctx.obj['fft_freq_range']))) + ' Hz'
    
    click.secho(f"\n -> Analysis completed: {freq_range} and file saved in:'{ctx.obj['search_path']}'.\n", fg = 'green', bold = True)

### ------------------------------ VERIFY PSDs ---------------------------------- ###      
@main.command()
@click.pass_context
def verify(ctx):
    """
    Manual verification of PSDs
    """
    import matplotlib.pyplot as plt
    from select_psd import matplotGui
    from pick import pick
    
    # check if index file was not found
    if not ctx.obj['power_present']:
        click.secho(f"\n -> File '{ctx.obj['file_power_mat']}' was not found in '{ctx.obj['search_path']}'.\n", fg = 'yellow', bold = True)
        return
    
    # select from command list
    main_dropdown_list = ['Original', 'Verified']
    title = 'Load File:'
    option, index = pick(main_dropdown_list, title, indicator = '-> ')
    
    if option == 'Original': # load index and power
        index_df = pd.read_csv(ctx.obj['index_path'])
        power_df = pd.read_pickle(ctx.obj['power_mat_path'])
    elif option == 'Verified':
        index_df = pd.read_csv(ctx.obj['index_verified_path'])
        power_df = pd.read_pickle(ctx.obj['power_mat_verified_path'])

    # init gui object
    callback = matplotGui(ctx.obj, index_df, power_df)
    plt.subplots_adjust(bottom=0.15) # create space for buttons
    
    # add title and labels
    callback.fig.suptitle('Select PSDs', fontsize=12)        # title
    callback.fig.text(0.5, 0.09,'Frequency (Hz)', ha="center")                                          # xlabel
    callback.fig.text(.02, .5, 'Power (V^2/Hz)', ha='center', va='center', rotation='vertical')         # ylabel
    callback.fig.text(0.9, 0.04,'**** KEY ** Previous : <-, Next: ->, Accept: Y, Reject: N ****' ,      # move/accept labels
                      ha="right", bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9),))              
                                                    
    # add key press
    callback.fig.canvas.mpl_connect('key_press_event', callback.keypress)
    plt.show()
 
### ------------------------------ PLOT ---------------------------------- ###     
@main.command()
@click.option('--freq', type = str, help = 'Enter frequency range, e.g. 1-30')
@click.pass_context
def plot(ctx, freq):
    """Enter plot menu"""
    
    from psd_analysis import melted_power_area, melted_power_ratio, melted_psds, plot_mean_psds
    from pick import pick
    from facet_plot_gui import GridGraph
    
    # check if power mat exists
    if not ctx.obj['power_present']:
        click.secho(f"\n -> File '{ctx.obj['file_power_mat']}' was not found in '{ctx.obj['search_path']}'" + 
                    "Need to run 'stft' before plotting.\n", fg = 'yellow', bold = True)
        return
        
    # select from command list
    main_dropdown_list = ['mean PSDs','summary plot and data export - (power area)',
                          'summary plot and data export - (power ratio)']
    title = 'Please select file for analysis: '
    option, index = pick(main_dropdown_list, title, indicator = '-> ')
    
    # load index and power mat
    index_df = pd.read_csv(ctx.obj['index_verified_path'])
    power_df = pd.read_pickle(ctx.obj['power_mat_verified_path'])
    
    # get categories
    categories = list(index_df.columns[index_df.columns.get_loc('stop_time')+1:])
    click.echo(categories)
    if option == 'summary plot and data export - (power area)':
        
        # get power area
        data = melted_power_area(index_df, power_df, ctx.obj['freq_ranges'], categories)
        
        # Graph interactive summary plot
        GridGraph(ctx.obj['search_path'], ctx.obj['melted_power_mat'], data).draw_graph(ctx.obj['summary_plot_type'])
        return
    
    if option == 'summary plot and data export - (power ratio)':
        
        # get power ratio
        data = melted_power_ratio(index_df, power_df,  ctx.obj['freq_ratios'], categories)
        
        # Graph interactive summary plot
        GridGraph(ctx.obj['search_path'], ctx.obj['melted_power_mat'], data).draw_graph(ctx.obj['summary_plot_type'])
        return
    
    # get frequency
    if freq is not None:
        freq_range = [int(i) for i in freq.split('-')]
        if len(freq_range) !=2:
              click.secho(f"\n -> '{freq}' could not be parsed. Please use the following format: 1-30.\n", fg = 'yellow', bold = True)
              return
    else:
        click.secho("\n -> 'Missing argument 'freq'. Please use the following format: --freq 1-30.\n", fg = 'yellow', bold = True)
        return
    
    if option == 'mean PSDs':
        df = melted_psds(index_df, power_df, freq_range, categories)
        plot_mean_psds(df, categories)
    
    

# Execute if module runs as main program
if __name__ == '__main__': 
    
    # define settings path
    settings_path = 'settings.yaml'
    # start
    main(obj={})
    
    
    
    