# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sakeplot.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SAKEDSP(object):
    def setupUi(self, SAKEDSP):
        SAKEDSP.setObjectName("SAKEDSP")
        SAKEDSP.resize(326, 270)
        self.centralwidget = QtWidgets.QWidget(SAKEDSP)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verifyButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.verifyButton.setFont(font)
        self.verifyButton.setObjectName("verifyButton")
        self.gridLayout.addWidget(self.verifyButton, 2, 7, 1, 1)
        self.PSDButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.PSDButton.setFont(font)
        self.PSDButton.setObjectName("PSDButton")
        self.gridLayout.addWidget(self.PSDButton, 3, 7, 1, 1)
        self.pathButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pathButton.setFont(font)
        self.pathButton.setObjectName("pathButton")
        self.gridLayout.addWidget(self.pathButton, 0, 7, 1, 1)
        self.PowerAreaButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.PowerAreaButton.setFont(font)
        self.PowerAreaButton.setObjectName("PowerAreaButton")
        self.gridLayout.addWidget(self.PowerAreaButton, 4, 7, 1, 1)
        self.errorBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.errorBrowser.setFont(font)
        self.errorBrowser.setObjectName("errorBrowser")
        self.gridLayout.addWidget(self.errorBrowser, 6, 0, 1, 8)
        self.STFTButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.STFTButton.setFont(font)
        self.STFTButton.setObjectName("STFTButton")
        self.gridLayout.addWidget(self.STFTButton, 1, 0, 1, 8)
        self.labelPSDRange = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.labelPSDRange.setFont(font)
        self.labelPSDRange.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelPSDRange.setObjectName("labelPSDRange")
        self.gridLayout.addWidget(self.labelPSDRange, 3, 0, 1, 2)
        self.labelPlotType = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.labelPlotType.setFont(font)
        self.labelPlotType.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelPlotType.setObjectName("labelPlotType")
        self.gridLayout.addWidget(self.labelPlotType, 4, 1, 1, 1)
        self.labelThresh = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.labelThresh.setFont(font)
        self.labelThresh.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelThresh.setObjectName("labelThresh")
        self.gridLayout.addWidget(self.labelThresh, 2, 0, 1, 2)
        self.PSDEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.PSDEdit.setFont(font)
        self.PSDEdit.setFrame(True)
        self.PSDEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.PSDEdit.setObjectName("PSDEdit")
        self.gridLayout.addWidget(self.PSDEdit, 3, 2, 1, 5)
        self.threshEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.threshEdit.setFont(font)
        self.threshEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.threshEdit.setObjectName("threshEdit")
        self.gridLayout.addWidget(self.threshEdit, 2, 2, 1, 5)
        self.pathEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pathEdit.setFont(font)
        self.pathEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.pathEdit.setReadOnly(True)
        self.pathEdit.setObjectName("pathEdit")
        self.gridLayout.addWidget(self.pathEdit, 0, 0, 1, 7)
        self.plotType = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.plotType.setFont(font)
        self.plotType.setObjectName("plotType")
        self.gridLayout.addWidget(self.plotType, 4, 6, 1, 1)
        self.plotValue = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.plotValue.setFont(font)
        self.plotValue.setObjectName("plotValue")
        self.gridLayout.addWidget(self.plotValue, 4, 2, 1, 4)
        SAKEDSP.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SAKEDSP)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 326, 18))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        SAKEDSP.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(SAKEDSP)
        self.statusbar.setObjectName("statusbar")
        SAKEDSP.setStatusBar(self.statusbar)
        self.actionSettings = QtWidgets.QAction(SAKEDSP)
        self.actionSettings.setObjectName("actionSettings")
        self.menuFile.addAction(self.actionSettings)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(SAKEDSP)
        QtCore.QMetaObject.connectSlotsByName(SAKEDSP)

    def retranslateUi(self, SAKEDSP):
        _translate = QtCore.QCoreApplication.translate
        SAKEDSP.setWindowTitle(_translate("SAKEDSP", "SAKE Plot"))
        self.verifyButton.setText(_translate("SAKEDSP", "Verify"))
        self.PSDButton.setText(_translate("SAKEDSP", "Plot PSD"))
        self.pathButton.setText(_translate("SAKEDSP", "Set Path..."))
        self.PowerAreaButton.setText(_translate("SAKEDSP", "Plot Power"))
        self.STFTButton.setText(_translate("SAKEDSP", "Fourier Transform (STFT)"))
        self.labelPSDRange.setText(_translate("SAKEDSP", "PSD Range (hz):"))
        self.labelPlotType.setText(_translate("SAKEDSP", "Power Area Type:"))
        self.labelThresh.setText(_translate("SAKEDSP", "Outlier Threshold:"))
        self.PSDEdit.setText(_translate("SAKEDSP", "1-30"))
        self.threshEdit.setText(_translate("SAKEDSP", "4"))
        self.menuFile.setTitle(_translate("SAKEDSP", "File"))
        self.actionSettings.setText(_translate("SAKEDSP", "Settings"))
