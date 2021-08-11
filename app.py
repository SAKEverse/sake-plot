# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:10:37 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import os
import click
from pick import pick 
########## ---------------------------------------------------------------- ##########



@click.group()
@click.option('--run_stft', default='1-30', help = 'Run stft analysis')
@click.pass_context
def run_stft(run_stft):

    # freq_range = input('Enter Frequency range (Hz): e.g.(5-10):\n')
    click.secho(f"\n -> '{run_stft}' was chosen\n" , fg = 'green', bold = True)


@click.command()
@click.argument('csv_path', type = str)
@click.pass_context
def main(csv_path:str):
    """

    ██████  ██    ██ ██████  ███████ ██████                                     
    ██   ██  ██  ██  ██   ██ ██      ██   ██                                      
    ██████    ████   ██   ██ ███████ ██████                                                                            
    ██         ██    ██   ██      ██ ██                                           
    ██         ██    ██████  ███████ ██                                           
             
     

    Inputs
    
    
    -----------
    
    
    csv_path : str, path to index file



    """
    
    
    
    if not os.path.exists(os.path.dirname(csv_path)):
        click.secho(f"\n -> '{csv_path}' path was not found\n" , fg = 'yellow', bold = True)
        return
    
    # select from command list
    main_dropdown_list = ['PSD', 'individual PSDs', 'summary plot', '']
    title = 'Please select file for analysis: '
    option, index = pick(main_dropdown_list, title, indicator = '-> ')
    
    

# Execute if module runs as main program
if __name__ == '__main__': 
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
