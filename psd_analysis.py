########## ------------------------------- IMPORTS ------------------------ ##########
import numpy as np
import pandas as pd
from stft import Stft, get_freq_index, Properties
from scipy.stats import gaussian_kde
from get_data import AdiGet
from tqdm import tqdm
from beartype import beartype
from typing import TypeVar
PandasDf = TypeVar('pandas.core.frame.DataFrame')
########## ---------------------------------------------------------------- ##########

@beartype
def get_power_area(pmat:np.ndarray, freq_vec:np.ndarray, freqs:np.ndarray) -> np.ndarray:
    """
    Get power area across frequencies

    Parameters
    ----------
    pmat : np.ndarray, 2D array containing power values rows = frequency bins and cols = time bins
    freq_vec : np.ndarray, vector with real frequency values
    freqs : np.ndarray, 2D array with frequencies, rows = different frequency ranges, colums = [start, stop]

    Returns
    -------
    powers : 1D np.array, len = frequency ranges 

    """
    
    # get frequency index
    freqs = freqs.reshape([-1, 2]) # reshape to 2D
    freq_idx = get_freq_index(freq_vec, freqs)
    
    # init empty array to store powers
    powers = np.zeros(freq_idx.shape[0])
    for i in range(freq_idx.shape[0]):
        powers[i] = np.sum(pmat[freq_idx[i,0]:freq_idx[i,1],:])
    
    return powers

@beartype
def get_power_ratio(pmat:np.ndarray, freq_vec:np.ndarray, freqs:np.ndarray) -> np.ndarray:
    """
    Get power ratio across frequencies

    Parameters
    ----------
    pmat : np.ndarray, 2D array containing power values rows = frequency bins and cols = time bins
    freq_vec : np.ndarray, vector with real frequency values
    freqs : np.ndarray, 3D array with frequencies, 1d = frequency ratios, 2d [lower to upper freq range], 3d = [start, stop]

    Returns
    -------
    powers : 1D np.array, len = frequency ranges 

    """
    
    # get frequency index
    freqs = freqs.reshape([-1,2,2]) # reshape to 3D
    freq_idx = get_freq_index(freq_vec, freqs)
    
    # init empty array to store powers
    powers = np.zeros(freq_idx.shape[0])
    for i in range(freq_idx.shape[0]):
        powers[i] = np.divide(np.sum(pmat[freq_idx[i,0,0]:freq_idx[i,0,1],:]),
                              np.sum(pmat[freq_idx[i,1,0]:freq_idx[i,1,1],:]))                           
    return powers
        
@beartype
def get_pmat(index_df:PandasDf, properties:dict) -> tuple((PandasDf, PandasDf)):
    """
    Run Stft analysis on signals retrieved using rows of index_df.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    properties: Dict

    Returns
    -------
    index_df: PandasDf, experiment index
    power_df: PandasDf, with power matrix and frequency

    """
    
    # drop rows containing NaNs
    index_df = index_df.dropna().reset_index()
    index_df = index_df.drop(['index'], axis = 1)

    # create empty series dataframe
    df = pd.DataFrame(np.empty((len(index_df), 2)), columns = ['freq', 'pmat'], dtype = object)

    for i in tqdm(range(len(index_df))): # iterate over dataframe
        
        # get properties
        file_properties = index_df[AdiGet.input_parameters].loc[i].to_dict()
        
        # get signal
        signal = AdiGet(file_properties).get_data_adi()
        
        # add sampling rate
        properties.update({'sampling_rate':int(file_properties['sampling_rate'])})
        
        # Init Stft object with required properties
        selected_keys = Properties.types.keys()
        selected_keys = list(selected_keys)
        
        # select key-value pairs from dictionary
        selected_properties = {x: properties[x] for x in selected_keys}
        
        # convert time series to frequency domain
        stft_obj = Stft(selected_properties)
        df.at[i, 'freq'], df.at[i, 'pmat'] = stft_obj.run_stft(signal)
    
    return index_df, df


def melted_power_area(index_df:PandasDf, power_df:PandasDf, freqs:list, selected_categories:list):
    """
    Get power area and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted

    Returns
    -------
    df : PandasDf, melted df with power area and categories

    """
    # convert to numpy array
    freqs = np.array(freqs)
    
    # create frequency column names
    freq_columns = []
    for i in range(freqs.shape[0]):
        freq_columns.append(str(freqs[i,0]) + ' - ' + str(freqs[i,1]) + ' Hz')
    
    # create array for storage
    power_array = np.empty((len(index_df), freqs.shape[0]))
    for i in range(len(index_df)): # iterate over dataframe
        
        # get power across frequencies
        power_array[i,:] = get_power_area(power_df['pmat'][i], power_df['freq'][i], freqs)
        
    # concatenate to array
    index_df = pd.concat([index_df, pd.DataFrame(data = power_array, columns = freq_columns)], axis=1)
    
    # set file id as index
    index_df.set_index('file_id', inplace = True)    
    
    # melt dataframe for seaborn plotting
    df = pd.melt(index_df, id_vars = selected_categories, value_vars = freq_columns, var_name = 'freq', value_name = 'power_area',
                 ignore_index = False)
    
    return df

def melted_power_ratio(index_df:PandasDf, power_df:PandasDf, freqs:list, selected_categories:list):
    """
    Get power ratio and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted

    Returns
    -------
    df : PandasDf, melted df with power area and categories

    """
    # convert to numpy array
    freqs = np.array(freqs)
    
    # create frequency column names
    freq_columns = []
    for i in range(freqs.shape[0]):
        freq_columns.append(str(freqs[i,0]) + ' - ' + str(freqs[i,1]) + ' Hz')
    
    # create array for storage
    power_array = np.empty((len(index_df), freqs.shape[0]))
    for i in range(len(index_df)): # iterate over dataframe
        
            # get power across frequencies
            power_array[i,:] = get_power_ratio(power_df['pmat'][i], power_df['freq'][i], freqs)
            
    # concatenate to array
    index_df = pd.concat([index_df, pd.DataFrame(data = power_array, columns = freq_columns)], axis=1)
    
    # set file id as index
    index_df.set_index('file_id', inplace = True)  
        
    # melt dataframe for seaborn plotting
    df = pd.melt(index_df, id_vars = selected_categories, value_vars = freq_columns, var_name = 'freq', value_name = 'power_ratio',
                 ignore_index=False)
    
    return df


def melted_psds(index_df:PandasDf, power_df:PandasDf, freq_range:list, selected_categories:list): ## don't drop file index
    """
    Get PSD and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted

    Returns
    -------
    df : PandasDf, melted df with psd and categories

    """

    # create arrays for storage
    power_array = np.array([])
    freq_array = np.array([])
    repeat_array = np.zeros(len(index_df))
    
    # get selected columns
    df = index_df[['file_id'] + selected_categories]
    
    for i in range(len(index_df)): # iterate over dataframe
        
        # unpack frequency and power
        freq = power_df['freq'][i]
        power = power_df['pmat'][i]
        
        # get desired frequency index
        f_idx = get_freq_index(freq, freq_range)
        freq = freq[f_idx[0]:f_idx[1]+1]
        power = np.mean(power[f_idx[0]:f_idx[1]+1,:], axis =1)
        
        # append to array
        power_array = np.concatenate((power_array, power))
        freq_array = np.concatenate((freq_array, freq ))
        
        # get length
        repeat_array[i] = freq.shape[0]

    # repeat array
    df = df.reindex(df.index.repeat(repeat_array))
    
    # set file id as index
    df.set_index('file_id', inplace = True) 
    
    # append to dataframe
    df['freq'] = freq_array
    df['power'] = power_array

    return df

def melted_power_dist(index_df:PandasDf, power_df:PandasDf, freq_range:list, selected_categories:list): ## don't drop file index
    """
    Get Power distribution and melt dataframe for seaborn plotting.

    Parameters
    ----------
    index_df : PandasDf, experiment index
    power_df : PandasDf, contains pmat and frequency vectors for every row of index_df
    freqs : list, 2D list with frequency ranges for extraction of power area
    selected_categories : list, columns that will be included in the melted

    Returns
    -------
    df : PandasDf, melted df with pdf and categories

    """

    # create arrays for storage
    power_array = np.array([])
    density_array = np.array([])
    threshold_array = np.array([])
    repeat_array = np.zeros(len(index_df))
    
    # get selected columns
    df = index_df[['file_id'] + selected_categories]
    
    # get all power areas
    for i in range(len(index_df)): # iterate over dataframe
        
        # unpack frequency and power
        freq = power_df['freq'][i]
        power = power_df['pmat'][i]
        
        # get desired frequency index
        f_idx = get_freq_index(freq, freq_range)
        freq = freq[f_idx[0]:f_idx[1]+1]
        power =  np.mean(power[f_idx[0]:f_idx[1]+1,:], axis = 0)
        
        power_df.at[i, 'pmat'] = power
        
        # append to array
        power_array = np.concatenate((power_array, power))
    
    # get mean and sdev for normalization
    avg = np.mean(power_array)
    sdev = np.std(power_array)
    
    # define edges for z normalized data and preallocate power_array
    power_array = np.array([])
    
    edges = np.linspace(-5, 5, 100)
    for i in range(len(index_df)):    
        
        # normalize
        power = (power_df['pmat'][i] - avg )/ sdev
        
        # select power above threshold
        threshold =  np.mean(power) + np.std(power)
        # power = power[power > threshold]
        
        # get kde
        pdf = gaussian_kde(power, bw_method = 0.5).evaluate(edges)
        
        # append to array
        power_array = np.concatenate((power_array, edges))
        density_array = np.concatenate((density_array, pdf))
        threshold_array = np.concatenate((threshold_array, np.repeat(threshold, power.shape[0])))
        
        # get length
        repeat_array[i] = edges.shape[0]

    # repeat array
    df = df.reindex(df.index.repeat(repeat_array))
    
    # set file id as index
    df.set_index('file_id', inplace = True) 
    
    # append to dataframe
    # df['threshold'] = threshold_array
    df['power'] = power_array
    df['density'] = density_array

    return df

if __name__ == '__main__':
    x = 1
    import os, yaml
    from load_index import load_index
    # from facet_plot_gui import GridGraph
    
    ### ---------------------- USER INPUT -------------------------------- ###
    
    ## define path and conditions for filtering
    filename = 'index.csv'
    parent_folder = r'C:\Users\panton01\Desktop\example_files'
    path =  os.path.join(parent_folder, filename)
    
    ## enter filter conditions
    # filter_conditions = {'brain_region':['bla', 'pfc'], 'treatment':['baseline','vehicle']} #
    
    ## define frequencies of interest
    with open('settings.yaml', 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)
    
    ### ---------------------------------------------------------------- ####
    
    ## load data frame
    index_df = load_index(path)
    
    ## save dataframe
    # index_df.to_csv(os.path.join(parent_folder, filename.replace('csv','pickle')))
    
    # get pmat
        # get power 
    # power_df = get_pmat(index_df, settings)
    
    # power_df.to_pickle(os.path.join(parent_folder, 'power_mat.pickle'))
    power_df = pd.read_pickle(r'C:\Users\panton01\Desktop\example_files\power_mat.pickle')
    df = melted_power_dist(index_df, power_df, [30,70], ['sex', 'treatment', 'brain_region'])
    
    # # remove mains noise and outliers!!!!!!!!!!!!!!!!!!!!! 
    # df = melted_power_ratio(index_df, power_df, settings['freq_ratios'], ['sex', 'treatment', 'brain_region']) #
    
    # import seaborn as sns
    
    # # get melted power area
    # df = melted_power_area(index_df, power_df, settings['freq_ranges'], ['sex', 'treatment', 'brain_region'])
    # # sns.catplot(data = df, x = 'freq', y = 'power_area', hue = 'treatment', col = 'sex', row = 'brain_region', kind = 'box')
    
    # # get melted psd
    # df = melted_psds(index_df, power_df, [1,30], ['sex', 'treatment', 'brain_region'])
    # df.to_csv('melted_psd.csv',index=False)
    # # g = sns.FacetGrid(df.iloc[::5,:], hue='treatment', row='sex', col='brain_region', palette='plasma')
    # # g.map(sns.lineplot, 'freq', 'power')
    
    # df.to_csv('melted_psd.csv',index=True)
    # path = r'C:\Users\panton01\Desktop\pydsp_analysis'
    # filename = 'power_area_df.csv'
    # df.to_csv(os.path.join(path, filename), index = False)
    
    # graph = GridGraph(path, filename)
    # graph.draw_graph('violin')











