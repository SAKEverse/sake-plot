# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:04:21 2021

@author: SuperComputer1
"""
from sakeplot_ui import Ui_SAKEDSP
import sys, os
from PyQt5 import QtCore, QtWidgets, QtTest 
from PyQt5.QtGui import QPixmap
import yaml
import subprocess
import webbrowser
from verify_psd import matplotGui
from psd_analysis import melted_power_area, melted_power_ratio, melted_psds
from facet_plot_gui import GridGraph
from psd_analysis import get_pmat



app = QtWidgets.QApplication(sys.argv)
Dialog = QtWidgets.QMainWindow()
ui = Ui_SAKEDSP()
ui.setupUi(Dialog)
Dialog.show()
_translate = QtCore.QCoreApplication.translate
script_dir=os.path.dirname(os.path.realpath(__file__))


ui.plotType.addItems(['box','violin','boxen','bar'])
ui.plotValue.addItems(['Area','Ratio'])

class ctx():
    obj={}

def updateImage(path):
    ui.picLabel.setPixmap(QPixmap(path))
    ui.picLabel.setScaledContents(True)
    

def setpath():
    """Set path to index file parent directory"""    
    # add path to original ctx.obj
    widget=QtWidgets.QFileDialog()
    path=widget.getExistingDirectory(None,_translate("SAKEDSP", 'Set path for index.csv'),ctx.obj['search_path'])
    ui.pathEdit.setText(_translate("SAKEDSP",path))
    msg=subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "setpath", path],capture_output=True)
    ui.errorBrowser.setText(_translate("SAKEDSP",str(msg.stdout.decode())))
    get_current_img()
    
ui.pathButton.clicked.connect(lambda:setpath())

def stft():
    
    updateImage(os.path.join(script_dir,r"images\bomb2.png"))
    
    ui.errorBrowser.setText(_translate("SAKEDSP","Processing... Check Terminal for Progess Bar"))
    
    QtTest.QTest.qWait(100)
    msg=subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "stft"])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("SAKEDSP","ERROR: Could not perform STFT... \nCheck terminal for errors..."))
        return

    updateImage(os.path.join(script_dir,r"images\bomb3.png"))
    
    ui.errorBrowser.setText(_translate("SAKEDSP",'STFT Complete!'))
    
    
ui.STFTButton.clicked.connect(lambda:stft())

def plotPSD():
    """Enter plot menu"""
    freq_range= ui.PSDEdit.text()
    msg=subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "plot", "--freq", freq_range, '--plot_type','psd'])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("SAKEDSP","Check Terminal for Errors"))

ui.PSDButton.clicked.connect(lambda:plotPSD())

def plotPower():
    if ui.plotValue.currentText() == 'Area':
        msg=subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "plot", '--plot_type','power_area','--kind',ui.plotType.currentText()])
    
    if ui.plotValue.currentText() == 'Ratio':
        msg=subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "plot", '--plot_type','power_ratio','--kind',ui.plotType.currentText()])
        
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("SAKEDSP","Check Terminal for Errors"))
    

    
ui.PowerAreaButton.clicked.connect(lambda:plotPower())

def plotDist():
    """Enter plot menu"""
    freq_range= ui.distEdit.text()
    msg=subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "plot", "--freq", freq_range, '--plot_type','dist'])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("SAKEDSP","Check Terminal for Errors"))
    

    
ui.distButton.clicked.connect(lambda:plotDist())

def verify():
    """
    Manual verification of PSDs
    """
    
    #update threshold
    threshold= ui.threshEdit.text()

    msg=subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "verify", "--outlier_threshold", threshold])
    
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("SAKEDSP",'ERROR: Unable to verify... \nCheck terminal for errors'))
        return
    
    ui.errorBrowser.setText(_translate("SAKEDSP",'Verified!'))
    updateImage(os.path.join(script_dir,r"images\bomb4.png"))
    
    
ui.verifyButton.clicked.connect(lambda:verify())

def reverify():
    """
    Manual verification of PSDs
    """
    
    #update threshold
    threshold= ui.threshEdit.text()

    msg=subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "verify", "--outlier_threshold", threshold,"--option","re"])
    print()
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("SAKEDSP",'ERROR: Unable to reverify... \nCheck terminal for errors'))
        return
    
    ui.errorBrowser.setText(_translate("SAKEDSP",'Reverified!'))
    updateImage(os.path.join(script_dir,r"images\bomb4.png"))
    
ui.reverifyButton.clicked.connect(lambda:reverify())

def openSettings():
    webbrowser.open(os.path.join(script_dir,r"settings.yaml"))
    
ui.actionSettings.triggered.connect(lambda:openSettings())

def get_current_img():
    subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "checkpath"])
    settings_path = 'settings.yaml'
    
    with open(settings_path, 'r') as file:
        ctx.obj = yaml.load(file, Loader=yaml.FullLoader)
        
    if ctx.obj['power_verified_present'] == 1:
        img=r"images\bomb4.png"
    elif ctx.obj['power_present'] == 1:
        img=r"images\bomb3.png"
    elif ctx.obj['index_present'] == 1:
        img=r"images\bomb2.png"
    else:
        img=r"images\bomb1.png"
        
    updateImage(os.path.join(script_dir,img))

# Execute if module runs as main program
if __name__ == '__main__': 

    # define ctx.obj path
    settings_path = 'settings.yaml'
    
    with open(settings_path, 'r') as file:
        ctx.obj = yaml.load(file, Loader=yaml.FullLoader)
    
    ui.pathEdit.setText(_translate("SAKEDSP", ctx.obj['search_path']))
    ui.threshEdit.setText(_translate("SAKEDSP", str(ctx.obj['outlier_threshold'])))
    
    
    get_current_img()
    app.exec_()
    
    
    
    
    
    
    
    