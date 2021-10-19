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
    ctx.obj.update({'power_verified_present':0})
    
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
        
        # check if power mat file is present and get full path
    if os.path.isfile(ctx.obj['power_mat_verified_path']):
        ctx.obj.update({'power_verified_present':1})
    
            
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
    index_df, power_df = get_pmat(index_df, ctx.obj)
    
    # save index and power
    index_df.to_csv(ctx.obj['index_path'], index = False)
    power_df.to_pickle(ctx.obj['power_mat_path'])
    
    # get freq_range for display
    freq_range = '-'.join(list(map(str, ctx.obj['fft_freq_range']))) + ' Hz'
    
    click.secho(f"\n -> Analysis completed: {freq_range} and file saved in:'{ctx.obj['search_path']}'.\n", fg = 'green', bold = True)

### ------------------------------ VERIFY PSDs ---------------------------------- ###      
@main.command()
@click.option('--outlier_threshold', type = str, help = 'Enter outlier threshold, e.g. 4.5')
@click.pass_context
def verify(ctx, outlier_threshold):
    """
    Manual verification of PSDs
    """
    from verify_psd import matplotGui
    
    # check if index file was not found
    if not ctx.obj['power_present']:
        click.secho(f"\n -> File '{ctx.obj['file_power_mat']}' was not found in '{ctx.obj['search_path']}'.\n", fg = 'yellow', bold = True)
        return
    
    # pass threshold to properties dictionary
    if outlier_threshold is None:
        click.secho("\n -> 'Missing argument 'outlier_threshold'. Please use the following format: --outlier_threshold 4.2.\n", fg = 'yellow', bold = True)
        return
    else:    
        ctx.obj.update({'outlier_threshold': float(outlier_threshold)})
    
    # load files
    index_df = pd.read_csv(ctx.obj['index_path'])
    power_df = pd.read_pickle(ctx.obj['power_mat_path'])
    
    # init gui object
    matplotGui(ctx.obj, index_df, power_df)


### ------------------------------ VERIFY PSDs ---------------------------------- ###      
@main.command()
@click.option('--outlier_threshold', type = str, help = 'Enter outlier threshold, e.g. 4.5')
@click.pass_context
def verifyr(ctx, outlier_threshold):
    """
    Manual re-verification of PSDs
    """
    from verify_psd import matplotGui
    
    # check if index file was not found
    if not ctx.obj['power_verified_present']:
        click.secho(f"\n -> File '{ctx.obj['file_power_mat_verified']}' was not found in '{ctx.obj['search_path']}'.\n", fg = 'yellow', bold = True)
        return
    
    # pass threshold to properties dictionary
    if outlier_threshold is None:
        click.secho("\n -> 'Missing argument 'outlier_threshold'. Please use the following format: --outlier_threshold 4.2.\n", fg = 'yellow', bold = True)
        return
    else:    
        ctx.obj.update({'outlier_threshold': float(outlier_threshold)})
    
    # load files
    index_df = pd.read_csv(ctx.obj['index_verified_path'])
    power_df = pd.read_pickle(ctx.obj['power_mat_verified_path'])

    # init gui object
    matplotGui(ctx.obj, index_df, power_df)

 
### ------------------------------ PLOT ---------------------------------- ###     
@main.command()
@click.option('--plot_type', type = str, help = 'Enter plot type, e.g. psd')
@click.option('--freq', type = str, help = 'Enter frequency range, e.g. 1-30')
@click.pass_context
def plot(ctx, freq, plot_type):
    """Enter plot menu"""
    
    from psd_analysis import melted_power_area, melted_power_ratio, melted_psds
    from facet_plot_gui import GridGraph
    
    # check if power mat exists
    if not ctx.obj['power_present']:
        click.secho(f"\n -> File '{ctx.obj['file_power_mat']}' was not found in '{ctx.obj['search_path']}'" + 
                    "Need to run 'stft' before plotting.\n", fg = 'yellow', bold = True)
        return
    
    # check if plot type is present in the correct format
    plot_type_options = ['power_area', 'power_ratio', 'psd']
    
    if plot_type is None:
        click.secho("\n -> 'Missing argument 'plot_type'. Please use the following format: --plot_type power_area.\n"
                    , fg = 'yellow', bold = True)
        return
    
    if plot_type not in plot_type_options:
        click.secho(f"\n -> Got'{plot_type}' instead of {plot_type_options}\n",
                    fg = 'yellow', bold = True)
        return
        
    # load index and power mat
    index_df = pd.read_csv(ctx.obj['index_verified_path'])
    power_df = pd.read_pickle(ctx.obj['power_mat_verified_path'])
    
    # get categories
    categories = list(index_df.columns[index_df.columns.get_loc('stop_time')+1:])
    
    if plot_type == 'power_area':
        
        # get power area
        data = melted_power_area(index_df, power_df, ctx.obj['freq_ranges'], categories)
        
        # Graph interactive summary plot
        GridGraph(ctx.obj['search_path'], ctx.obj['power_mat_verified_path'], data).draw_graph(ctx.obj['summary_plot_type'])
        return
    
    if plot_type == 'power_ratio':
        
        # get power ratio
        data = melted_power_ratio(index_df, power_df,  ctx.obj['freq_ratios'], categories)
        
        # Graph interactive summary plot
        GridGraph(ctx.obj['search_path'], ctx.obj['power_mat_verified_path'], data).draw_graph(ctx.obj['summary_plot_type'])
        return
    
    if plot_type == 'psd':
        
        # get frequency
        if freq is not None:
            freq_range = [int(i) for i in freq.split('-')]
            if len(freq_range) !=2:
                  click.secho(f"\n -> '{freq}' could not be parsed. Please use the following format: 1-30.\n", fg = 'yellow', bold = True)
                  return
        else:
            click.secho("\n -> 'Missing argument 'freq'. Please use the following format: --freq 1-30.\n", fg = 'yellow', bold = True)
            return
    
        # get psd data
        psd_data = melted_psds(index_df, power_df, freq_range, categories)
        
        # Graph interactive PSD
        GridGraph(ctx.obj['search_path'],  ctx.obj['psd_mat'], psd_data).draw_psd()
        
# Execute if module runs as main program
if __name__ == '__main__': 
    
    # define settings path
    settings_path = 'settings.yaml'
    # start
    main(obj={})
    
    
    
    