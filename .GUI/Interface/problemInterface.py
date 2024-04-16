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
from .Compoent import BannerWidget,ToggleButton

class ProblemInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.view = QWidget(self);self.view.setObjectName("MainView")
        self.vBoxLayout = QVBoxLayout(self.view)
        
        self.__initLayout()
        self.__initWidget()
    def __initLayout(self):
        self.bannerWidget=BannerWidget(self)
        self.bannerWidget.setTitle("Problem Definition")
        self.bannerWidget.setPixmap('./picture/header1.png')
        self.bannerWidget.addWidget(SegmentedInterface(self))
        
        self.vBoxLayout.addWidget(self.bannerWidget)
        self.vBoxLayout.setContentsMargins(0,0,0,0)
        
    def __initWidget(self):
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        with open("./qss/problem_interface.qss") as f:
            self.setStyleSheet(f.read())
            
class PivotInterface(QWidget):
    """ Pivot interface """
    Nav = Pivot
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.pivot = self.Nav(self)
        self.stackedWidget = QStackedWidget(self)
        self.vBoxLayout = QVBoxLayout(self)

        self.VariableInterface = VariableInterface(self)
        self.DriverInterface = QLabel('Model Driver', self)

        self.addSubInterface(self.VariableInterface, 'VariableInterface', self.tr('Decision Variables'))
        self.addSubInterface(self.DriverInterface, 'DriverInterface', self.tr('Model Driver'))
        
        self.vBoxLayout.addWidget(self.pivot, 0, Qt.AlignLeft)
        self.vBoxLayout.addWidget(self.stackedWidget)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)

        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.VariableInterface)
        self.pivot.setCurrentItem(self.VariableInterface.objectName())
    
        qrouter.setDefaultRouteKey(self.stackedWidget, self.VariableInterface.objectName())
    def addSubInterface(self, widget: QLabel, objectName, text):
        widget.setObjectName(objectName)
        # widget.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget)
        )
    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())
        qrouter.push(self.stackedWidget, widget.objectName())
class SegmentedInterface(PivotInterface):
    '''SegmentedInterface'''
    Nav = SegmentedWidget
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vBoxLayout.removeWidget(self.pivot)
        self.vBoxLayout.insertWidget(0, self.pivot)
        self.vBoxLayout.setContentsMargins(28,15,0,0)
        self.pivot.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)      
        for item in self.pivot.items.values():
            item.setMinimumWidth(180)
            with open("./qss/pivot.qss") as f:
                item.setStyleSheet(f.read())
                
class VariableInterface(ScrollArea): 
    """ Variable Interface-1"""
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.view=QWidget(self)
        self.view.setObjectName("VariableView")
        self.vBoxLayout=QVBoxLayout(self.view) #Main
        self.hTopBoxLayout=QHBoxLayout() #
           
        self.tableFrame=TableFrame(self)
        self.builtInParaCard=BuiltInParaCard(self,tableFrame=self.tableFrame)
        self.userParaCard=UserParaCard(self,tableFrame=self.tableFrame)
        
        self.__initLayout()
        self.__initWidget()

    def __initWidget(self):
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        
    def __initLayout(self):
        self.vBoxLayout.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.vBoxLayout.setContentsMargins(0,0,0,0)
        
        self.hTopBoxLayout.addWidget(self.builtInParaCard)
        self.hTopBoxLayout.addWidget(self.userParaCard)
        
        self.vBoxLayout.addLayout(self.hTopBoxLayout)
        self.vBoxLayout.addWidget(self.tableFrame)
            
class BuiltInParaCard(QFrame):
    pillButtons=[]
    def __init__(self, parent=None,tableFrame=None):
        super().__init__(parent)
        self.setObjectName("BuiltInParaCard")
        self.tableFrame=tableFrame
        
        
        self.subtitle=StrongBodyLabel("Built-in Parameter Card", self)
        self.ContentWidget=QFrame(self);self.ContentWidget.setObjectName("ContentWidget")
        self.FlowWidget=QFrame(self);self.FlowWidget.setObjectName("FlowWidget")
        self.vBoxLayout=QVBoxLayout(self)
        self.flowLayout=FlowLayout(self.FlowWidget)
        
        self.__initFlowWidget()
        self.__initContentWidget()
        self.__initLayout()
        self.__initWidget()
    def __initFlowWidget(self):
        self.flowLayout.setContentsMargins(5,5,5,5)
        self.flowLayout.setVerticalSpacing(20)
        self.flowLayout.setHorizontalSpacing(10)
    
    def __initContentWidget(self): 
        self.ContentWidget.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Maximum)
        self.ContentLayout=QVBoxLayout(self.ContentWidget)
        self.ContentLayout.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        
        self.combobox=ComboBox(self)
        self.combobox.addItems(["SAC","SWAT","HEC-RAS","WRF"])
        self.combobox.setMinimumWidth(170);self.combobox.setCurrentIndex(0)
        
        hBoxLayout=QHBoxLayout()
        hBoxLayout.addWidget(BodyLabel("Built-in Models:"));hBoxLayout.addWidget(self.combobox)
        
        ###TODO
        Infos = [['UZTWM','10.00','300.00',"Uniform"],
                     ['UZFWM','5.00','150.00','Uniform'],
                     ['UZK','0.10','0.75','Uniform'],
                     ['PCTIM','0.00','0.10','Uniform'],
                     ['ADIMP','0.00','0.20','Uniform']      
        ]
        
        self.ContentLayout.addLayout(hBoxLayout)
        self.__generateFlowWidget(Infos)
    def __generateFlowWidget(self,texts):
        for i in reversed(range(self.flowLayout.count())): 
            self.flowLayout.itemAt(i).widget().setParent(None)
            
        for text in texts:
            pillButton=ToggleButton(text[0])
            pillButton.Content=text;pillButton.tableWidget=self.tableFrame
            self.pillButtons.append(pillButton)
            self.flowLayout.addWidget(pillButton)
    def __initWidget(self):
        self.subtitle.setContentsMargins(5,5,0,0)
    def __initLayout(self):
        self.vBoxLayout.setAlignment(Qt.AlignLeft|Qt.AlignTop)
        self.vBoxLayout.setContentsMargins(0,0,0,0)
        self.vBoxLayout.addWidget(self.subtitle, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.vBoxLayout.addWidget(self.ContentWidget)
        self.vBoxLayout.addWidget(self.FlowWidget)
        
class UserParaCard(QFrame):
    tableFrame=None
    def __init__(self, parent=None,tableFrame=None):
        super().__init__(parent)
        self.setObjectName("UserParaCard")
        self.tableFrame=tableFrame
        
        self.vBoxLayout=QVBoxLayout(self)   
        
        self.subtitle=StrongBodyLabel("User-define Parameter Card",self)
        self.ContentWidget=QFrame(self);self.ContentWidget.setObjectName("ContentWidget")
      
        self.__initContentWidget()
        self.__initLayout()
        self.__initWidget()
        
        self.button1.clicked.connect(self.addItemToTable)
        self.button2.clicked.connect(self.clearAll)
        
    def addItemToTable(self):
        table=self.tableFrame
        name=self.lineEdit1.text()
        lowBound=self.lineEdit2.text()
        uperBound=self.lineEdit3.text()
        distribution=self.combobox2.text()
        
        info=[name,lowBound,uperBound,distribution]
        rows=table.table.rowCount()
        table.table.insertRow(rows)
        table.addRows(info,rows)
        self.clearAll()
        
    def clearAll(self):
        self.lineEdit1.clear()
        self.lineEdit2.setValue(0.00)
        self.lineEdit3.setValue(0.00)
        self.combobox2.setCurrentIndex(0)

    def __initContentWidget(self):
        self.ContentWidget.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Maximum)       
        self.ContentLayout=QVBoxLayout(self.ContentWidget)
        
        vBoxLayout1=QVBoxLayout();vBoxLayout2=QVBoxLayout()
        self.label1=BodyLabel("Parameter Name:")
        vBoxLayout1.addWidget(self.label1)
        self.lineEdit1=LineEdit();self.lineEdit1.setClearButtonEnabled(True);self.lineEdit1.setMaximumWidth(1000)
        font=QFont();font.setFamily('Segoe UI');font.setPointSize(12)
        self.lineEdit1.setFont(font)
        vBoxLayout2.addWidget(self.lineEdit1)
        
        self.label2=BodyLabel("Lower Bound:");self.label2.setAlignment(Qt.AlignmentFlag.AlignRight)
        vBoxLayout1.addWidget(self.label2)
        self.lineEdit2=DoubleSpinBox();self.lineEdit2.setRange(-1000,1000);self.lineEdit2.setMaximumWidth(1000)
        self.lineEdit2.setFont(font)
        vBoxLayout2.addWidget(self.lineEdit2)
        
        self.label3=BodyLabel("Upper Bound:");self.label3.setAlignment(Qt.AlignmentFlag.AlignRight)
        vBoxLayout1.addWidget(self.label3)
        self.lineEdit3=DoubleSpinBox();self.lineEdit3.setRange(-1000,1000);self.lineEdit3.setMaximumWidth(1000)
        self.lineEdit3.setFont(font)
        vBoxLayout2.addWidget(self.lineEdit3)
        vBoxLayout1.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.label4=BodyLabel("Distribution:");self.label4.setAlignment(Qt.AlignmentFlag.AlignRight)
        vBoxLayout1.addWidget(self.label4)
        self.combobox2=ComboBox();self.combobox2.addItems(["Uniform","Normal","LogNormal","Logistic"]);self.combobox2.setFont(font)
        vBoxLayout2.addWidget(self.combobox2)
        vBoxLayout1.setSpacing(19.5)
        
        self.button1=PushButton("Add");self.button1.setIcon(FluentIcon.ADD)
        self.button2=PushButton("Reset");self.button2.setIcon(FluentIcon.CLEAR_SELECTION)
        hBoxLayout=QHBoxLayout();hBoxLayout.addWidget(self.button1);hBoxLayout.addWidget(self.button2)    
        vBoxLayout2.addLayout(hBoxLayout)
        vBoxLayout1.addStretch(1)
        
        hBoxLayout=QHBoxLayout()
        hBoxLayout.addLayout(vBoxLayout1)
        hBoxLayout.addLayout(vBoxLayout2)
        
        TempWidget=QWidget();TempWidget.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        hBoxLayout.addWidget(TempWidget)
        
        self.ContentLayout.addLayout(hBoxLayout)
        self.ContentLayout.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
    def __initWidget(self):
        self.subtitle.setContentsMargins(5,5,0,0)
    def __initLayout(self):
        self.vBoxLayout.setAlignment(Qt.AlignHCenter|Qt.AlignTop)
        self.vBoxLayout.setContentsMargins(0,0,0,0)
        self.vBoxLayout.addWidget(self.subtitle, alignment=Qt.AlignmentFlag.AlignCenter)
        self.vBoxLayout.addWidget(self.ContentWidget)        

class Frame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=parent)   
        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setObjectName('frame')
        
        self.title=StrongBodyLabel("Parameter Summary List")
        self.topFrame=QFrame()
        self.topFrame.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Minimum)
        vboxLayout=QVBoxLayout(self.topFrame)
        vboxLayout.addWidget(self.title)
        self.vBoxLayout.addWidget(self.topFrame,alignment=Qt.AlignmentFlag.AlignHCenter)
        
    def addWidget(self, widget):
        self.vBoxLayout.addWidget(widget)
class TableWidgetWithDelete(TableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pressedButtons=[]
        self.inDel=False
        self.indDel=None
    def _setSelectedRows(self, indexes: List[QModelIndex]):
        if not self.inDel:
            if self.pressedButtons:
                for butt in self.pressedButtons:
                    butt.setVisible(False)
                self.pressedButtons=[]
            S=set()
            for index in indexes:
                S.add(index.row())
            for row in list(S):
                Widget=self.cellWidget(row,self.columnCount()-1)
                button=Widget.findChild(QPushButton)
                button.setVisible(True)
                self.pressedButtons.append(button)
            return super()._setSelectedRows(indexes)
class TableFrame(Frame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0,0,0,0)
        self.table = TableWidgetWithDelete(self)
        self.addWidget(self.table)

        self.infos={}
        self.builtInfos={}
        self.table.verticalHeader().hide()
        self.table.setColumnCount(6)
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderLabels([self.tr(' '),
            self.tr('Parameter'), self.tr('Lower Bound'), self.tr('Upper Bound'),
            self.tr('Distribution'),self.tr('Operation')])
        Infos=[]
        self.counts=len(Infos)
        for i, Info in enumerate(Infos):
            self.infos[i]=Info
            self.addRows(Info,i)
        self.setObjectName("ParameterSummary")
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
        self.table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Minimum)

        self.bottomFrame=QFrame()
        self.bottomFrame.setObjectName("bottomFrame")
        vBoxLayout=QVBoxLayout(self.bottomFrame)
        self.generateButton=PushButton("Generate Parameter File")
        self.loadButton=PushButton("Load Parameter File")
        hBoxLayout=QHBoxLayout()
        hBoxLayout.addStretch(1)
        hBoxLayout.addWidget(self.loadButton);hBoxLayout.addStretch(1);hBoxLayout.addWidget(self.generateButton)
        hBoxLayout.addStretch(1)
        hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        vBoxLayout.addLayout(hBoxLayout)
        self.addWidget(self.bottomFrame)
    def addRows(self,infos,row):
        self.infos[row]=infos
        columnCount=self.table.columnCount()
        
        item=QTableWidgetItem(str(row+1));item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
        item.setFlags(Qt.ItemFlag.ItemIsSelectable)
        self.table.setItem(row, 0, item)
        self.table.setCellWidget(row,self.table.columnCount()-1,self.__addDelButton())
        
        for col, info in zip(range(1,columnCount-1),infos):
            item=QTableWidgetItem(info);item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()
        self.table.selectRow(row)
        self.table.viewport().update()
        
    def __addDelButton(self):
        buttonWidget=QWidget();buttonWidget.setFixedHeight(50)
        buttonWidget.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        tempLayout=QHBoxLayout(buttonWidget)
        tempLayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        deleteButton=PushButton("Delete",buttonWidget)
        deleteButton.setObjectName("tableButton")
        deleteButton.clicked.connect(self.deleteItem)
        deleteButton.setVisible(False)
        tempLayout.addWidget(deleteButton)
        return buttonWidget
    def deleteItem(self):
        selectedRows=list(self.table.delegate.selectedRows)
        for _ in selectedRows:
            self.table.inDel=True
            text=self.table.item(selectedRows[0],1).text()
            self.table.removeRow(selectedRows[0])
            if  text in self.builtInfos and not self.builtInfos[text].selfToggled:
                self.builtInfos[text].Status=0
                self.builtInfos[text].click()
        self.table.inDel=False
        self.table.delegate.selectedRows.clear()
        self.table.selectRow(selectedRows[0])
        del self.infos[selectedRows[0]]
        for row in range(selectedRows[0],self.table.rowCount()):
            self.table.item(row,0).setText(str(row+1))
            self.infos[row]= self.infos[row+1]
            self.infos.pop(row+1)
            

        
            
