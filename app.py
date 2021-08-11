# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:10:37 2021

@author: panton01
"""

########## ------------------------------- IMPORTS ------------------------ ##########
import os
import yaml
import click
from pick import pick 
########## ---------------------------------------------------------------- ##########


@click.group()
@click.pass_context
def main(ctx):
    """

    ██████  ██    ██ ██████  ███████ ██████                                     
    ██   ██  ██  ██  ██   ██ ██      ██   ██                                      
    ██████    ████   ██   ██ ███████ ██████                                                                            
    ██         ██    ██   ██      ██ ██                                           
    ██         ██    ██████  ███████ ██                                           
             
     
    """
    
    # read file
    with open(settings_path, 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)
        ctx.obj.update({'settings':settings})
        

@main.command()
@click.argument('path', type = str)
@click.pass_context
def setpath(ctx, path):
        
    # check if path exists
    if not os.path.isfile(path):
        click.secho(f"\n -> '{path}' file was not found.\n", fg = 'yellow', bold = True)
        return
    
    # set path
    ctx.obj['settings']['index_path'] = path
        
    # write to file
    with open(settings_path, 'w') as file:
        yaml.dump(ctx.obj['settings'], file)
        
    click.secho(f"\n -> Path was set to:'{path}'.\n", fg = 'green', bold = True)

@main.command()
@click.argument('freq_range', type = str)
# @click.option('--stft', default='1-30', help = 'Run stft analysis')
@click.pass_context
def stft(freq_range):

    click.secho(f"\n -> '{freq_range}' was chosen\n" , fg = 'green', bold = True)
    
    
@main.command()
@click.argument('freq_range', type = str)
@click.pass_context
def plot(freq_range):

    click.secho(f"\n -> '{freq_range}' was chosen\n" , fg = 'green', bold = True)
    # select from command list
    main_dropdown_list = ['PSD', 'individual PSDs', 'summary plot', '']
    title = 'Please select file for analysis: '
    option, index = pick(main_dropdown_list, title, indicator = '-> ')
    
    

# Execute if module runs as main program
if __name__ == '__main__': 
    
    # define settings path
    settings_path = 'settings.yaml'
    # start
    main(obj={})
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
