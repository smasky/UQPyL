from typing import List, Optional
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QColor, QPixmap,QPainter, QBrush, QPainterPath,QFont
from PySide6.QtWidgets import  QLabel,QFrame,QSizePolicy
from PySide6.QtCore import Qt,  QSize,QRectF,QPointF
from PySide6.QtWidgets import QHeaderView,QWidget,QTableWidgetItem,QVBoxLayout,QPushButton,QHBoxLayout,QLabel,QStackedWidget,QFrame
from PySide6.QtGui import QColor, QPixmap,QColor,QLinearGradient,QGradient,QBrush
from qfluentwidgets import (Pivot, qrouter, SegmentedWidget,ComboBox,StrongBodyLabel, BodyLabel,ScrollArea,
                            ComboBox,TableWidget,FluentIcon,PushButton,StrongBodyLabel,LineEdit,DoubleSpinBox,
                            PillPushButton,FlowLayout)
from qfluentwidgets import FluentIcon

class ToggleButton(PillPushButton):
    delButton=None
    Index=-1
    tableWidget=None
    isToggled=False
    selfToggled=False
    Content=None
    Name=None
    Count=1
    Status=1
    def __init__(self,parent=None):
        super().__init__(parent)
        self.clicked.connect(self.__toggleOperation)
        
    def setTableWidget(self,widget):
        self.tableWidget=widget
        
    def __toggleOperation(self):
        if(self.Count % 2):
            self.isToggled=not self.isToggled
            if(self.isToggled):
                self.Count=1
                self.Status=1
                self.__addItem()
            else:
                self.selfToggled=True
                self.__deleteItem()
                self.selfToggled=False
        self.Count+=1   
    def __addItem(self):
        self.Name=self.Content[0]
        rowCount=self.tableWidget.table.rowCount()
        self.tableWidget.table.insertRow(rowCount)
        self.tableWidget.addRows(self.Content,rowCount)
        self.Index=rowCount
        self.delButton=self.tableWidget.table.cellWidget(rowCount,self.tableWidget.table.columnCount()-1).findChild(QPushButton)
        self.tableWidget.builtInfos[self.Name]=self
    
    def __deleteItem(self):
        if (self.Status):
            self.tableWidget.table.delegate.selectedRows.clear()
            infos=self.tableWidget.infos
            for k, v in infos.items():
                if v[0] == self.Name:
                    self.Index=k
                    break
            self.tableWidget.table.selectRow(self.Index)
            self.delButton.click()
        
        
    
    
        
        