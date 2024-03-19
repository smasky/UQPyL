from typing import List, Optional
from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QColor, QPixmap,QPainter, QBrush, QPainterPath,QFont
from PySide6.QtWidgets import  QLabel,QFrame,QSizePolicy, QFileDialog,QPushButton,QButtonGroup
from PySide6.QtCore import Qt,  QSize,QRectF,QPointF
from PySide6.QtWidgets import QHeaderView,QWidget,QTableWidgetItem,QVBoxLayout,QPushButton,QHBoxLayout,QLabel,QStackedWidget,QFrame
from PySide6.QtGui import QColor, QPixmap,QColor,QLinearGradient,QGradient,QBrush
from qfluentwidgets import (Pivot, qrouter, SegmentedWidget,ComboBox,StrongBodyLabel, BodyLabel,ScrollArea,
                            ComboBox,TableWidget,FluentIcon,PushButton,StrongBodyLabel,LineEdit,DoubleSpinBox,
                            PillPushButton,FlowLayout)
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import ConfigItem,FolderValidator,RadioButton,SpinBox
from .Compoent import BannerWidget, PushSettingCard
class DOEInterface(ScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view=QWidget(self)
        self.vBoxLayout=QVBoxLayout(self.view)
        self.banner=BannerWidget(self)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.__initBanner()
        self.__initLayout()
        self.__initWidget()
        
    def __initBanner(self):
        self.banner.setTitle("Design of Experiment")
        self.banner.setPixmap("./picture/header3.png")
    def __initLayout(self):
        self.vBoxLayout.addWidget(self.banner)    
        self.banner.setContentsMargins(28,0,0,0)
        
    def __initWidget(self):
        self.view.setObjectName('view')
        self.setObjectName('DOEInterface')
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.banner)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        
        self.banner.addWidget(ModelInfoCard(self))
        self.banner.addWidget(MethodCard(self))
        
        with open("./qss/DOE_interface.qss") as f:
            self.setStyleSheet(f.read())
class MethodCard(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.vBoxLayout=QVBoxLayout(self)   
        self.setObjectName("MethodCard")
        self.subtitle=StrongBodyLabel("Design of Experiment Method", self)
        self.ContentWidget=QFrame(self)
        self.ContentWidget.setObjectName("ContentWidget")
        self.stackWidget=QStackedWidget(self.ContentWidget)
        self.stackWidget.addWidget(LatinHypercube())
        self.stackWidget.addWidget(StackBaseWidget())
        self.stackWidget.addWidget(QusiMonteCarlo())
        self.stackWidget.addWidget(StackBaseWidget())
        self.stackWidget.addWidget(StackTipWidget(tipsText="Please enter the number of discrete levels for each factor in the design, separated by comma.", text="Number of levels"))
        self.stackWidget.addWidget(StackTipWidget(tipsText="Number of Factors should be equal as number of parameters in parameter file.", text="Number of Factors"))
        self.stackWidget.addWidget(StackTipWidget(tipsText="Number of Factors should be equal as number of parameters in parameter file.", text="Number of Factors"))
        self.stackWidget.addWidget(StackTipWidget(tipsText="Number of Factors should be equal as number of parameters in parameter file.", text="Number of Factors"))
        self.stackWidget.addWidget(StackTipWidget(tipsText="Number of Factors should be equal as number of parameters in parameter file.", text="Number of Factors"))
        self.stackWidget.addWidget(StackTipWidget(tipsText="Number of Factors should be equal as number of parameters in parameter file.", text="Number of Factors"))
        self.stackWidget.addWidget(StackTipWidget(tipsText="Number of total samples points= (dimension+1)*Number of Trajectories", text="Number of Trajectories"))
        self.stackWidget.addWidget(StackTipWidget(tipsText="Number of total samples points= (dimension+2)*Number of Base sequence", text="Number of Base sequence"))
        self.stackWidget.addWidget(StackTipWidget(tipsText="Number of total samples points= dimension*Number of Base Samples", text="Number of Base Samples"))
        self.stackWidget.addWidget(StackTipWidget(tipsText="Number of total samples points= (dimension+1)*Number of Base Samples", text="Number of Base Samples"))

        self.stackWidget.setObjectName("StackWidget")
     
        self.__initContentWidget()
        self.__initLayout()
        self.__initWidget()
    def __initContentWidget(self):
        vBoxLayout=QVBoxLayout(self.ContentWidget)
        vBoxLayout.setContentsMargins(0,10,0,0)
        self.ContentWidget.setContentsMargins(0,0,0,0)
        hBoxLayout=QHBoxLayout();hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hBoxLayout.setContentsMargins(10,0,0,0)
        hBoxLayout.addWidget(BodyLabel("Choose DoE Method:"))
        self.combox=ComboBox();self.combox.setFixedWidth(250)
        hBoxLayout.addWidget(self.combox)
        self.combox.addItems(["Latin Hypercube","Monte Carlo","Quasi-Monte Carlo","Good Lattice Point","General Full-Factorial",
                              "2-Level Full-Factorial","2-Level Fractional Factorial",
                              "Plackett-Burman","Box-Behnken","Central-Composite",
                              "Morris One at A Time","Sobol Sampling","FAST Sampling",
                              "Finite Difference"])
        vBoxLayout.addLayout(hBoxLayout)
        vBoxLayout.addWidget(self.stackWidget)
        vBoxLayout.addWidget(PushButton("Generate Sampling Points"),alignment=Qt.AlignmentFlag.AlignHCenter)
        vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.combox.currentIndexChanged.connect(self.stackWidget.setCurrentIndex)
        
    def __initLayout(self):    
        self.vBoxLayout.addWidget(self.subtitle,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.vBoxLayout.addWidget(self.ContentWidget)
    def __initWidget(self):
        self.setContentsMargins(0,0,0,0)
class StackBaseWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vBoxLayout=QVBoxLayout(self)
        self.setContentsMargins(0,0,0,0)
        hBoxLayout1=QHBoxLayout();hBoxLayout1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hBoxLayout1.addWidget(BodyLabel("Number of sample points:"))
        spinBox=SpinBox()
        spinBox.setRange(0,10000)
        spinBox.setMinimumWidth(250)
        hBoxLayout1.setContentsMargins(92,0,0,0)
        hBoxLayout1.setSpacing(50)
        hBoxLayout1.addWidget(spinBox) 
              
        self.vBoxLayout.addSpacing(75)
        self.vBoxLayout.addLayout(hBoxLayout1)
        self.vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setObjectName("LatinHypercubeWidget")
        # self.setStyleSheet("background-color:red;")

class StackTipWidget(QWidget):
    def __init__(self, tipsText="",text="",parent=None):
        super().__init__(parent)
        self.vBoxLayout=QVBoxLayout(self)
        
        hBoxLayout=QHBoxLayout();hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hBoxLayout.addWidget(BodyLabel("Tip:"))
        # tips=BodyLabel("Please enter the number of discrete levels for each factor in the design, separated by comma.")
        tips=BodyLabel(tipsText)
        tips.setStyleSheet("color:rgb(73, 73, 73);")
        hBoxLayout.addWidget(tips)
        
        self.setContentsMargins(0,0,0,0)
        hBoxLayout1=QHBoxLayout();hBoxLayout1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        hBoxLayout1.addWidget(BodyLabel(text))
        self.spinBox=SpinBox()
        self.spinBox.setRange(0,10000)
        self.spinBox.setMinimumWidth(250)
        hBoxLayout1.setContentsMargins(92,0,0,0)
        hBoxLayout1.setSpacing(50)
        hBoxLayout1.addWidget(self.spinBox) 
              
        self.vBoxLayout.addLayout(hBoxLayout)
        self.vBoxLayout.addSpacing(50)
        self.vBoxLayout.addLayout(hBoxLayout1)
        self.vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setObjectName("LatinHypercubeWidget")

class QusiMonteCarlo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vBoxLayout=QVBoxLayout(self)
        hBoxLayout=QHBoxLayout();hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hBoxLayout.setSpacing(50)
        hBoxLayout.addWidget(BodyLabel("Choose the type of MonteCarlo Method:"),alignment=Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignLeft)
        self.radioWidget=self.__initButtonGroup()
        hBoxLayout.addWidget(self.radioWidget)
        hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignLeft)
        self.setContentsMargins(0,0,0,0)
        hBoxLayout.setContentsMargins(0,0,0,0)
        
        self.vBoxLayout.addLayout(hBoxLayout)
        
        hBoxLayout1=QHBoxLayout();hBoxLayout1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hBoxLayout1.addWidget(BodyLabel("Number of sample points:"))
        spinBox=SpinBox()
        spinBox.setRange(0,10000)
        spinBox.setMinimumWidth(250)
        hBoxLayout1.setContentsMargins(92,0,0,0)
        hBoxLayout1.setSpacing(50)
        hBoxLayout1.addWidget(spinBox) 
              
        
        self.vBoxLayout.addLayout(hBoxLayout1)
        self.vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setObjectName("LatinHypercubeWidget")
        # self.setStyleSheet("background-color:red;")
    def __initButtonGroup(self):
        radioWidget = QWidget()
        radioLayout = QVBoxLayout(radioWidget)
        radioLayout.setContentsMargins(0, 0, 0, 0)
        radioLayout.setSpacing(15)
        radioButton1 = RadioButton(self.tr('QMC Sobol\'Sequence'), radioWidget)
        radioButton2 = RadioButton(self.tr('QMC Halton Sequence'), radioWidget)
        radioButton3 = RadioButton(self.tr('QMC Faure Sequence'), radioWidget)
        radioButton4 = RadioButton(self.tr('QMC Hammersle Sequence'), radioWidget)
        buttonGroup = QButtonGroup(radioWidget)
        buttonGroup.addButton(radioButton1)
        buttonGroup.addButton(radioButton2)
        buttonGroup.addButton(radioButton3)
        buttonGroup.addButton(radioButton4)
        radioLayout.addWidget(radioButton1)
        radioLayout.addWidget(radioButton2)
        radioLayout.addWidget(radioButton3)
        radioLayout.addWidget(radioButton4)
        radioButton1.click()
        
        return radioWidget
        
class LatinHypercube(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vBoxLayout=QVBoxLayout(self)
        hBoxLayout=QHBoxLayout();hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hBoxLayout.setSpacing(50)
        hBoxLayout.addWidget(BodyLabel("Choose the type of Latin Hypercube Method:"),alignment=Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignLeft)
        self.radioWidget=self.__initButtonGroup()
        hBoxLayout.addWidget(self.radioWidget)
        hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignLeft)
        self.setContentsMargins(0,0,0,0)
        hBoxLayout.setContentsMargins(0,0,0,0)
        
        self.vBoxLayout.addLayout(hBoxLayout)
        
        hBoxLayout1=QHBoxLayout();hBoxLayout1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hBoxLayout1.addWidget(BodyLabel("Number of sample points:"))
        spinBox=SpinBox()
        spinBox.setRange(0,10000)
        spinBox.setMinimumWidth(250)
        hBoxLayout1.setContentsMargins(120,0,0,0)
        hBoxLayout1.setSpacing(50)
        hBoxLayout1.addWidget(spinBox) 
              
        
        self.vBoxLayout.addLayout(hBoxLayout1)
        self.vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setObjectName("LatinHypercubeWidget")
        # self.setStyleSheet("background-color:red;")
    def __initButtonGroup(self):
        radioWidget = QWidget()
        radioLayout = QVBoxLayout(radioWidget)
        radioLayout.setContentsMargins(0, 0, 0, 0)
        radioLayout.setSpacing(15)
        radioButton1 = RadioButton(self.tr('Random Latin Hypercube Sampling'), radioWidget)
        radioButton2 = RadioButton(self.tr('Center Latin Hypercube Sampling'), radioWidget)
        radioButton3 = RadioButton(self.tr('Maximin Latin Hypercube Sampling'), radioWidget)
        radioButton4 = RadioButton(self.tr('Center Maximin Latin Hypercube Sampling'), radioWidget)
        radioButton5 = RadioButton(self.tr('Correlation Latin Hypercube Sampling'), radioWidget)
        radioButton6 = RadioButton(self.tr('Symmetric Latin Hypercube Sampling'), radioWidget)
        buttonGroup = QButtonGroup(radioWidget)
        buttonGroup.addButton(radioButton1)
        buttonGroup.addButton(radioButton2)
        buttonGroup.addButton(radioButton3)
        buttonGroup.addButton(radioButton4)
        buttonGroup.addButton(radioButton5)
        buttonGroup.addButton(radioButton6)
        radioLayout.addWidget(radioButton1)
        radioLayout.addWidget(radioButton2)
        radioLayout.addWidget(radioButton3)
        radioLayout.addWidget(radioButton4)
        radioLayout.addWidget(radioButton5)
        radioLayout.addWidget(radioButton6)
        radioButton1.click()
        
        return radioWidget
        
        
        
class ModelInfoCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent, )
        
        self.vBoxLayout=QVBoxLayout(self)   
        self.setObjectName("ModelInfoCard")
        self.subtitle=QLabel("Model Information", self)
        self.subtitle.setObjectName("settingLabel")
        self.ContentWidget=QFrame(self)
        self.ContentWidget.setObjectName("ContentWidget")
        
        self.__initContentWidget()
        self.__initLayout()
        self.__initWidget()
    def __initContentWidget(self):
        self.ContentWidget.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Maximum)       
        self.ContentLayout=QVBoxLayout(self.ContentWidget)
        self.ContentLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ContentLayout.setSpacing(5)
        self.ContentLayout.setContentsMargins(0,0,0,10)
        self.DecisionParaCard = PushSettingCard(
            self.tr('Choose file'),
            FIF.FOLDER,
            self.tr("Set the position of Decision Parameter File (.py)"),
            "Provide the position of decision parameter information file.",
            "Folder",
            self
        )
        self.ModelExecutableCard = PushSettingCard(
            self.tr('Choose file'),
            FIF.FOLDER,
            self.tr("Set the position of Model Executable Program (.exe)"),
            "Provide the position of the executable model.",
            "Folder",
            self
        )
        self.PostProcessCard = PushSettingCard(
            self.tr('Choose file'),
            FIF.FOLDER,
            self.tr("Set the position of Postprocessing File (.py)"),
            "Provide the position of postprocessing file.",
            "Folder",
            self
        )
        
        self.ContentLayout.addWidget(self.DecisionParaCard)
        self.ContentLayout.addWidget(self.ModelExecutableCard)
        self.ContentLayout.addWidget(self.PostProcessCard)
        
        self.DecisionParaCard.clicked.connect(self.__onClickedDecCard)
        self.ModelExecutableCard.clicked.connect(self.__onClickedModCard)
        self.PostProcessCard.clicked.connect(self.__onClickedPostCard)
    def __onClickedDecCard(self):
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Choose File"), "./","Python File *.py")  
        if folder[0]:
            self.DecisionParaCard.setContent(folder[0])
        else:
            self.DecisionParaCard.setContent("Please choose again!")
    def __onClickedModCard(self):
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Choose File"), "./","Executable Program *.exe") 
        if folder[0]:
            self.ModelExecutableCard.setContent(folder[0])
        else:
            self.ModelExecutableCard.setContent("Please choose again!")
    def __onClickedPostCard(self):
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Choose File"), "./","Python File *.py") 
        if folder[0]:
            self.PostProcessCard.setContent(folder[0])
        else:
            self.PostProcessCard.setContent("Please choose again!")
    def __initLayout(self):
        self.vBoxLayout.setAlignment(Qt.AlignHCenter|Qt.AlignTop)
        self.vBoxLayout.setContentsMargins(0,0,0,0)
        self.vBoxLayout.addWidget(self.subtitle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.vBoxLayout.addWidget(self.ContentWidget)  
    #    
    def __initWidget(self):
        self.subtitle.setContentsMargins(5,5,0,0)
