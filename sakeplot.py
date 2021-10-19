# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:04:21 2021

@author: SuperComputer1
"""
from sakeplot_ui import Ui_SAKEDSP
import sys, os
from PyQt5 import QtCore, QtWidgets
import yaml
import subprocess


app = QtWidgets.QApplication(sys.argv)
Dialog = QtWidgets.QMainWindow()
ui = Ui_SAKEDSP()
ui.setupUi(Dialog)
Dialog.show()
_translate = QtCore.QCoreApplication.translate
script_dir=os.path.dirname(os.path.realpath(__file__))


ui.plotType.addItems(['box','violin','boxen'])
ui.plotValue.addItems(['Area','Ratio'])

class ctx():
    obj={}

def setpath():
    """Set path to index file parent directory"""    
    # add path to original ctx.obj
    widget=QtWidgets.QFileDialog()
    path=widget.getExistingDirectory(None,_translate("SAKEDSP", 'Set path for index.csv'),ctx.obj['search_path'])
    subprocess.run(["python", os.path.join(script_dir,r"pydsp.py"), "setpath", path])
    
ui.pathButton.clicked.connect(lambda:setpath())

def stft():

    ui.errorBrowser.setText(_translate("SAKEDSP","Processing... Check Terminal for Progess Bar"))
    subprocess.run(["python", os.path.join(script_dir,r"pydsp.py"), "stft"])

    # get freq_range for display
    freq_range = '-'.join(list(map(str, ctx.obj['fft_freq_range']))) + ' Hz'
    ui.errorBrowser.setText(_translate("SAKEDSP",f"\n -> Analysis completed: {freq_range} and file saved in:'{ctx.obj['search_path']}'.\n"))
    
    
ui.STFTButton.clicked.connect(lambda:stft())

def plotPSD():
    """Enter plot menu"""
    freq_range= ui.PSDEdit.text()
    subprocess.run(["python", os.path.join(script_dir,r"pydsp.py"), "plot", "--freq", freq_range, '--plot_type','psd'])

ui.PSDButton.clicked.connect(lambda:plotPSD())

def plotPower():
    if ui.plotValue.currentText() == 'Area':
        subprocess.run(["python", os.path.join(script_dir,r"pydsp.py"), "plot", '--plot_type','power_area','--kind',ui.plotType.currentText()])
    
    if ui.plotValue.currentText() == 'Ratio':
        subprocess.run(["python", os.path.join(script_dir,r"pydsp.py"), "plot", '--plot_type','power_ratio','--kind',ui.plotType.currentText()])

ui.PowerAreaButton.clicked.connect(lambda:plotPower())

def verify():
    """
    Manual verification of PSDs
    """
    
    #update threshold
    threshold= ui.threshEdit.text()

    subprocess.run(["python", r"C:\Users\SuperComputer1\Documents\GitHub\pydsp\pydsp.py", "verify", "--outlier_threshold", threshold])
    
    
ui.verifyButton.clicked.connect(lambda:verify())

# Execute if module runs as main program
if __name__ == '__main__': 

    # define ctx.obj path
    settings_path = 'settings.yaml'
    
    with open(settings_path, 'r') as file:
        ctx.obj = yaml.load(file, Loader=yaml.FullLoader)
    
    ui.pathEdit.setText(_translate("SAKEDSP", ctx.obj['search_path']))
    ui.threshEdit.setText(_translate("SAKEDSP", str(ctx.obj['outlier_threshold'])))
    
    app.exec_()