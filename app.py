# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:10:37 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import os, yaml, click
from pick import pick
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
    
    # check if index file is present and get full path
    ctx.obj.update({'index_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_index'])})
    
    if os.path.isfile(ctx.obj['index_path']):
        ctx.obj.update({'index_present':1})
        ctx.obj.update({'index':load_index(ctx.obj['index_path'])})
        
    # check if power mat file is present and get full path
    ctx.obj.update({'power_mat_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_power_mat'])})
    if os.path.isfile(ctx.obj['power_mat_path']):
        ctx.obj.update({'power_present':1})
            
  
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
        
    # get power 
    power_df = get_pmat(ctx.obj['index'], fft_duration = ctx.obj['fft_win'],
                freq_range = ctx.obj['fft_freq_range'], f_noise = ctx.obj['mains_noise'])
    
    # save power
    power_df.to_pickle(ctx.obj['power_mat_path'])
    
    # get freq_range for display
    freq_range = '-'.join(list(map(str, ctx.obj['fft_freq_range']))) + ' Hz'
    
    click.secho(f"\n -> Analysis completed: {freq_range} and file saved in:'{ctx.obj['power_mat_path']}'.\n", fg = 'green', bold = True)
    
    
@main.command()
@click.option('--freq', type = str, help = 'Enter frequency range, e.g. 1-30')
@click.pass_context
def plot(ctx, freq):
    """Enter plot menu"""
    
    from psd_analysis import melted_power_area, melted_power_ratio, melted_psds, plot_mean_psds
    
    # check if index file exists
    if not ctx.obj['index_present']:
        click.secho(f"\n -> File '{ctx.obj['file_index']}' was not found in '{ctx.obj['search_path']}'.\n", fg = 'yellow', bold = True)
        return
    
    # check if power mat exists
    elif not ctx.obj['power_present']:
        click.secho(f"\n -> File '{ctx.obj['file_power_mat']}' was not found in '{ctx.obj['search_path']}'" + 
                    "Need to run 'stft' before plotting.\n", fg = 'yellow', bold = True)
        return
        
    # select from command list
    main_dropdown_list = ['mean PSDs', 'individual PSDs', 'summary plot and data export - (power area)',
                          'summary plot and data export - (power ratio)']
    title = 'Please select file for analysis: '
    option, index = pick(main_dropdown_list, title, indicator = '-> ')
    
    # load index and power mat
    index_df = pd.read_csv(ctx.obj['index_path'])
    power_df = pd.read_pickle(ctx.obj['power_mat_path'])
    
    # get categories
    categories = list(index_df.columns[index_df.columns.get_loc('stop_time')+1:])
    
    if option == 'summary plot and data export - (power area)':
        from facet_plot_gui import GridGraph
        
        # get power area
        data = melted_power_area(index_df, power_df, ctx.obj['freq_ranges'], categories)
        
        # Graph interactive summary plot
        GridGraph(ctx.obj['search_path'], ctx.obj['melted_power_mat'], data).draw_graph(ctx.obj['summary_plot_type'])
        return
    
    if option == 'summary plot and data export - (power ratio)':
        from facet_plot_gui import GridGraph
        
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
        df = melted_psds(index_df, power_df, freq_range, ['sex', 'treatment', 'brain_region'])
        plot_mean_psds(df)
    
    

# Execute if module runs as main program
if __name__ == '__main__': 
    
    # define settings path
    settings_path = 'settings.yaml'
    # start
    main(obj={})
    
    
    
    