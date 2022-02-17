# -*- coding: utf-8 -*-

### ------------------------- IMPORTS ----------------------------- ###
from sakeplot_ui import Ui_SAKEDSP
import sys, os
from PyQt5 import QtCore, QtWidgets, QtTest 
from PyQt5.QtGui import QPixmap
import subprocess
import webbrowser
import pandas as pd
from sakecli import main_check
### --------------------------------------------------------------------- ###


# init gui app
app = QtWidgets.QApplication(sys.argv)
Dialog = QtWidgets.QMainWindow()
ui = Ui_SAKEDSP()
ui.setupUi(Dialog)
Dialog.show()
_translate = QtCore.QCoreApplication.translate
script_dir=os.path.dirname(os.path.realpath(__file__))
ui.plotType.addItems(['box','violin','boxen','bar','swarm','strip'])
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

def norm_changed():
    """
    Manual verification of PSDs
    """
    
    if ui.checkBoxNorm.isChecked():
        subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "normalize", "--enable", "true", "--column", ui.normCol.currentText(), "--group", ui.normGroup.currentText()])
    else:
        subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "normalize", "--enable", "false", "--column", ui.normCol.currentText(), "--group", ui.normGroup.currentText()])
        
    
ui.checkBoxNorm.stateChanged.connect(lambda:norm_changed())

def norm_col_changed():
    """
    Manual verification of PSDs
    """
    try:
        index=pd.read_csv(os.path.join(ctx.obj['search_path'],'index.csv'))
        ui.normGroup.clear()
        ui.normGroup.addItems(index[ui.normCol.currentText()].unique())
        norm_changed()
    except:pass
        
    
ui.normCol.activated.connect(lambda:norm_col_changed())
ui.normGroup.activated.connect(lambda:norm_changed())

def openSettings():
    webbrowser.open(os.path.join(script_dir,r"settings.yaml"))
    
ui.actionSettings.triggered.connect(lambda:openSettings())

def get_current_img():
    subprocess.run(["python", os.path.join(script_dir,r"sakecli.py"), "checkpath"])
    
    ctx.obj = main_check(ctx)
    
    if os.path.isfile(ctx.obj['power_mat_verified_path']):
        img=r"images\bomb4.png"
    elif os.path.isfile(ctx.obj['power_mat_path']):
        img=r"images\bomb3.png"
    elif os.path.isfile(ctx.obj['index_path']):
        img=r"images\bomb2.png"
    else:
        img=r"images\bomb1.png"
        
    updateImage(os.path.join(script_dir,img))
    
    #get groups and columns
    try:
        ui.normCol.clear()
        ui.normGroup.clear()
        index=pd.read_csv(os.path.join(ctx.obj['search_path'],'index.csv'))
        ui.normCol.addItems(list(index.columns)[list(index.columns).index('stop_time')+1:-1])
        ui.normGroup.addItems(index[ui.normCol.currentText()].unique())
    except:pass

# Execute if module runs as main program
if __name__ == '__main__': 

    # define ctx.obj path
    ctx.obj = main_check(ctx)

    if not os.path.isdir(ctx.obj['search_path']):
        ctx.obj['search_path']=""

    ui.pathEdit.setText(_translate("SAKEDSP", ctx.obj['search_path']))
    ui.threshEdit.setText(_translate("SAKEDSP", str(ctx.obj['outlier_threshold'])))    
    ui.checkBoxNorm.setChecked(ctx.obj['normalize'])
    norm_changed()

    get_current_img()
    app.exec_()
    
    
    
    
    
    
    
    