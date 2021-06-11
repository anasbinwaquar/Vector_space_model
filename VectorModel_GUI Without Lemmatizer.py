# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TestUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSlider, QStatusBar, QVBoxLayout, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from withoutlemma import preprocessing, query_processing,document_matrix_creation
from PyQt5.QtCore import QCoreApplication, QMetaObject, QRect
import time
from PyQt5.QtGui import QFont
import operator

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        #Pre loading of data
        start = time.time()
        stopwords=[]
        inverted_index = {}
        tf_idf_index= {}
        preprocessing(stopwords,tf_idf_index)
        document_matrix_creation()
        end = time.time()
        # print
        MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(797, 639)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.run_query = QPushButton(self.centralwidget)
        self.run_query.setObjectName(u"run_query")
        self.run_query.setGeometry(QRect(570, 20, 121, 41))
        self.Query_input = QLineEdit(self.centralwidget)
        self.Query_input.setObjectName(u"Query_input")
        self.Query_input.setGeometry(QRect(120, 30, 411, 31))
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(230, 130, 251, 31))
        font = QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.scrollArea = QScrollArea(self.centralwidget)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setGeometry(QRect(240, 190, 151, 401))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 149, 399))
        self.verticalLayout = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.document_display = QLabel(self.scrollAreaWidgetContents)
        self.document_display.setObjectName(u"document_display")
        font1 = QFont()
        font1.setPointSize(12)
        self.document_display.setFont(font1)
        self.document_display.setWordWrap(True)

        self.verticalLayout.addWidget(self.document_display)

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(120, 80, 401, 31))
        font2 = QFont()
        font2.setPointSize(10)
        self.label.setFont(font2)
        self.time = QLabel(self.centralwidget)
        self.time.setObjectName(u"time")
        self.time.setGeometry(QRect(430, 230, 301, 51))
        font3 = QFont()
        font3.setPointSize(11)
        self.time.setFont(font3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
        _translate = QtCore.QCoreApplication.translate

        self.time.setText(_translate("MainWindow", 'Total Pre-Processing Time:'+str(end-start)))

    # setupUi

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "18K0198 Vector Scale Model Assignment 2"))
        self.run_query.setText(_translate("MainWindow", "Run Query"))
        #Button Function
        self.run_query.clicked.connect(self.process)

        # self.document_display.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", "Matched Documents"))
        # self.label.setText(QCoreApplication.translate("MainWindow", u"Query Format Example: t1 and t2 or t3 .. tn", None))
    
    def process(self):
        _translate = QtCore.QCoreApplication.translate
        query=self.Query_input.text()
        sting=''
        # print(type(query))
        # try:
        try:
            start = time.time()
            Ans=query_processing(query)
            Ans=dict( sorted(Ans.items(), key=operator.itemgetter(1),reverse=True))
            print(Ans)
            print(len(Ans))
            for A in Ans:
                sting+="Document: "+str(A)+" "
            end = time.time()
            TT=str(end-start)
            print(TT)
            if len(Ans)==0:
                self.document_display.setText(_translate("MainWindow", 'No Matched Documents'))
            else:
                self.document_display.setText(_translate("MainWindow", sting))
                self.time.setText(_translate("MainWindow", 'Total Query Time:'+TT))
        except(IndexError):
            self.document_display.setText(_translate("MainWindow", 'Incorrect Query'))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())