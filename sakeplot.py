# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:04:21 2021

@author: SuperComputer1
"""
from sakeplot_ui import Ui_SAKEDSP
import sys
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import yaml
import os
from filter_index import load_index
from psd_analysis import melted_psds
from facet_plot_gui import GridGraph
from psd_analysis import get_pmat


app = QtWidgets.QApplication(sys.argv)
Dialog = QtWidgets.QMainWindow()
ui = Ui_SAKEDSP()
ui.setupUi(Dialog)
Dialog.show()
_translate = QtCore.QCoreApplication.translate

class ctx():
    obj={}

def setpath():
    """Set path to index file parent directory"""    
    # add path to original ctx.obj
    widget=QtWidgets.QFileDialog()
    path=widget.getExistingDirectory(None,_translate("SAKEDSP", 'Set path for index.csv'),ctx.obj['search_path'])
    ctx.obj['search_path'] = path

    # write to file
    with open(settings_path, 'w') as file:
        yaml.dump(ctx.obj, file)
        
    out_str=f"\n -> Path was set to:'{path}'.\n"
    ui.pathEdit.setText(_translate("SAKEDSP", path))
    ui.errorBrowser.setText(_translate("SAKEDSP", out_str))
    
    if os.path.isfile(ctx.obj['index_path']):
        ctx.obj.update({'index_present':1})
    
ui.pathButton.clicked.connect(lambda:setpath())

def stft():
    """Runs the Short-time Fourier Transform"""


    # check if index file was not found
    if not ctx.obj['index_present']:
        ui.errorBrowser.setText(_translate("SAKEDSP",f"\n -> File '{ctx.obj['file_index']}' was not found in '{ctx.obj['search_path']}'.\n"))
        return

    # load index 
    index_df = load_index(ctx.obj['index_path'])
    
    ui.errorBrowser.setText(_translate("SAKEDSP","Analysis Go!"))
    # get power 
    power_df = get_pmat(index_df, ctx.obj)
    
    # save index and power
    index_df.to_csv(ctx.obj['index_verified_path'], index = False)
    power_df.to_pickle(ctx.obj['power_mat_path'])
    power_df.to_pickle(ctx.obj['power_mat_verified_path'])
    
    # get freq_range for display
    freq_range = '-'.join(list(map(str, ctx.obj['fft_freq_range']))) + ' Hz'
    
    ui.errorBrowser.setText(_translate("SAKEDSP",f"\n -> Analysis completed: {freq_range} and file saved in:'{ctx.obj['search_path']}'.\n"))
    
    # check if power mat file is present and get full path
    if os.path.isfile(ctx.obj['power_mat_path']):
        ctx.obj.update({'power_present':1})
    
ui.STFTButton.clicked.connect(lambda:stft())

def plotPSD():
    """Enter plot menu"""
    

    
    # check if power mat exists
    if not ctx.obj['power_present']:
        ui.errorBrowser.setText(_translate("SAKEDSP", f"\n -> File '{ctx.obj['file_power_mat']}' was not found in '{ctx.obj['search_path']}'" + 
                    "Need to run 'stft' before plotting.\n"))
        return
    
    # load index and power mat
    index_df = pd.read_csv(ctx.obj['index_verified_path'])
    power_df = pd.read_pickle(ctx.obj['power_mat_verified_path'])
    
    # get categories
    categories = list(index_df.columns[index_df.columns.get_loc('stop_time')+1:])
    ui.errorBrowser.setText(_translate("SAKEDSP", ','.join(categories)))
    
    # ERROR CHECKING FOR FREQUENCY RANGE (check the default valut in settings)

    
    freq_range= [int(ui.PSDEdit.text().split('-')[0]) ,int(ui.PSDEdit.text().split('-')[1])]
    breakpoint()
    # get psd data
    psd_data = melted_psds(index_df, power_df, freq_range, categories)
    
    # Graph interactive PSD
    GridGraph(ctx.obj['search_path'],  ctx.obj['psd_mat'], psd_data).draw_psd()

ui.PSDButton.clicked.connect(lambda:plotPSD())

# Execute if module runs as main program
if __name__ == '__main__': 

    # define ctx.obj path
    settings_path = 'settings.yaml'
    # start
    
    with open(settings_path, 'r') as file:
        ctx.obj = yaml.load(file, Loader=yaml.FullLoader)

    # set files present to zero
    ctx.obj.update({'index_present': 0})
    ctx.obj.update({'power_present': 0})
    
    # get path to files
    ctx.obj.update({'index_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_index'])})
    ctx.obj.update({'power_mat_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_power_mat'])})
    ctx.obj.update({'index_verified_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_index_verified'])})
    ctx.obj.update({'power_mat_verified_path': os.path.join(ctx.obj['search_path'], ctx.obj['file_power_mat_verified'])})
    
    ui.pathEdit.setText(_translate("SAKEDSP", ctx.obj['search_path']))
    
    # check if index file is present and get full path
    if os.path.isfile(ctx.obj['index_path']):
        ctx.obj.update({'index_present':1})
        
    # check if power mat file is present and get full path
    if os.path.isfile(ctx.obj['power_mat_path']):
        ctx.obj.update({'power_present':1})
    
    app.exec_()