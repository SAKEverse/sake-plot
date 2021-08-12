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
from app_interface import file_present_check
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
    
    # check if index file is present
    ctx.obj.update({'index_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_index'])})
    if os.path.isfile(ctx.obj['index_path']):
        ctx.obj.update({'index_present':1})
        
    # check if power mat file is present
    ctx.obj.update({'power_mat_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_power_mat'])})
    if os.path.isfile(ctx.obj['power_mat_path']):
        ctx.obj.update({'power_present':1})
        
  
@main.command()
@click.argument('path', type = str)
@click.pass_context
def setpath(ctx, path):
        
    # add path to original settings
    ctx.obj['settings']['search_path'] = path

    # write to file
    with open(settings_path, 'w') as file:
        yaml.dump(ctx.obj['settings'], file)
        
    click.secho(f"\n -> Path was set to:'{path}'.\n", fg = 'green', bold = True)


@main.command()
@click.argument('freq_range', type = str)
# @click.option('--stft', default='1-30', help = 'Run stft analysis')
@click.pass_context
def stft(ctx, freq_range):
    
    # check if path exists
    if not ctx.obj['index_present']:
        click.secho('\n -> Index file was not found.\n', fg = 'yellow', bold = True)
        return
    
    
    

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