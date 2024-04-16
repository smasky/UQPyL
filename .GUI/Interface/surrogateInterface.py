from typing import List, Optional
from PySide6.QtCore import QEasingCurve, Qt
from PySide6.QtGui import QColor, QPixmap,QPainter,QDoubleValidator,QIntValidator,QPen
from PySide6.QtWidgets import  QLabel,QFrame,QSizePolicy, QFileDialog
from PySide6.QtCore import Qt,QPointF,Signal
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QStackedWidget,QFrame
from PySide6.QtGui import QColor, QPixmap,QColor
from qfluentwidgets import (ComboBox,StrongBodyLabel, BodyLabel,ScrollArea,ComboBox,PushButton,StrongBodyLabel,LineEdit)
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import SwitchButton,OpacityAniStackedWidget,PopUpAniStackedWidget,FlowLayout,InfoBar,InfoBarPosition,InfoBarIcon
from .Compoent import BannerWidget,PushSettingCard,gpKernelCard,rbfnKernelCard,svmKernelCard,mlpKernelCard,StackWidget
paths={0:'GP-RBF.png',1:'GP-Matern.png',2:'GP-RQ.png',3:'GP-ESS.png',4:'GP-Dot.png',6:'RBF-Cubic.png',
       7:'RBF-Linear.png',8:'RBF-Multi.png',9:'RBF-Thin.png',10:'RBF-Gas.png',12:"SVM-Gas.png",13:"SVM-Poly.png",20:"KELM-Gas.png",21:"KELM-Linear.png",22:"KELM-Poly.png"}
Sizes={0:60,1:60,2:60,3:60,4:25,6:30,7:28,8:45,9:32,10:32,12:30,13:35,20:30,21:28,22:35}
NoImage=[5,11,14,15,16,17,18,19]

class surrogateInterface(ScrollArea):
    ModelIndex=0
    MethodIndex=0
    K_index=[0,6,12,14,17,20]
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view=QWidget(self)
        self.vBoxLayout=QVBoxLayout(self.view)
        self.view=BannerWidget(self)
        
        self.__initBanner()
        self.__initLayout()
        self.__initWidget()
    def __initBanner(self):
        self.view.setTitle("Surrogate Modelling")
        self.view.setPixmap("./picture/header4.png")
    def __initLayout(self): 
        self.view.addWidget(BasicInfoCard(self))
        self.methodCard=MethodCard(self)
        self.kernelSettingCard=KernelSettingCard()
        self.modelSettingCard=ModelSettingCard()
        self.kernelSettingCard.setBinder(self.methodCard)
        self.modelSettingCard.setBinder(self.methodCard)
        
        
        
        flowLayout=QHBoxLayout();flowLayout.addWidget(self.kernelSettingCard,4);flowLayout.addWidget(self.modelSettingCard,2)
        self.view.addWidget(self.methodCard)
        self.view.vBoxLayout.addLayout(flowLayout)
        self.view.setContentsMargins(28,0,0,0) #
        
        self.methodCard.BindStackLabel.connect(self.createTopRightInfoBar)
        self.methodCard.BindStackWidget.connect(self.setModelIndex)
    def __initWidget(self):
        self.setWidget(self.view)       
        self.vBoxLayout.setContentsMargins(0,0,0,0)#
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setObjectName('DOEInterface')
        with open("./qss/surrogate_interface.qss") as f:
            self.setStyleSheet(f.read())
    def setModelIndex(self,i):
        self.ModelIndex=i
    def createTopRightInfoBar(self,i):
        i=self.K_index[self.ModelIndex]+i
        if(i not in NoImage):
            info=InfoBar(icon=InfoBarIcon.INFORMATION,
                title=self.tr('Kernel Function:'),
                content=self.tr(' '),
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self
            )
            pixmap=QPixmap('./picture/Equations/'+paths[i])
            pixmap=pixmap.scaledToHeight(Sizes[i])
            info.contentLabel.setPixmap(pixmap)
            info.contentLabel.setFixedHeight(60)
            info.contentLabel.setVisible(True)
            info.show()
            
class BasicInfoCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("BasicInfoCard")
        self.vBoxLayout=QVBoxLayout(self)
        
        self.subtitle=QLabel("Basic Information", self);self.subtitle.setObjectName("BasicInfoLabel")
        self.ContentWidget=QFrame(self);self.ContentWidget.setObjectName("BasicInfoWidget")
        
        self.__initContentWidget()
        self.__initLayout()
        self.__initWidget()
    def __initContentWidget(self):
        self.ContentWidget.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Maximum)       
        self.ContentLayout=QVBoxLayout(self.ContentWidget)
        self.ContentLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ContentLayout.setSpacing(5)
        self.ContentLayout.setContentsMargins(0,0,0,10)
        self.DOECard = PushSettingCard(
            self.tr('Choose file'),
            FIF.FOLDER,
            self.tr("Set the position of DOE file (.doe)"),
            "Provide the position of decision parameter information file.",
            "Folder",
            self
        )
        self.ModelDriverCard = PushSettingCard(
            self.tr('Choose file'),
            FIF.FOLDER,
            self.tr("Set the position of Model Driver file (.mdr)"),
            "Provide the position of the executable model.",
            "Folder",
            self
        )
        self.ContentLayout.addWidget(self.DOECard)
        self.ContentLayout.addWidget(self.ModelDriverCard) 
        self.DOECard.clicked.connect(self.__onClickedDOECard)
        self.ModelDriverCard.clicked.connect(self.__onClickedModelDriverCard)
    def __onClickedDOECard(self):
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Choose File"), "./","DOE File *.doe")  
        if folder[0]:
            self.DOECard.setContent(folder[0])
        else:
            self.DOECard.setContent("Please choose again!")
    def __onClickedModelDriverCard(self):
        folder = QFileDialog.getOpenFileName(
            self, self.tr("Choose File"), "./","Model Driver File *.mdr") 
        if folder[0]:
            self.ModelDriverCard.setContent(folder[0])
        else:
            self.ModelDriverCard.setContent("Please choose again!")
    def __initLayout(self):
        self.vBoxLayout.setAlignment(Qt.AlignHCenter|Qt.AlignTop)
        self.vBoxLayout.setContentsMargins(0,0,0,0)
        self.vBoxLayout.addWidget(self.subtitle, alignment=Qt.AlignLeft)
        self.vBoxLayout.addWidget(self.ContentWidget)  
    def __initWidget(self):
        self.subtitle.setContentsMargins(5,5,0,0)
        
class MethodCard(QFrame):
    BindStackWidget=Signal(int)
    BindStackKernel=Signal(int)
    BindStackLabel=Signal(int)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("MethodCard")
        self.vBoxLayout=QVBoxLayout(self)   
        
        self.subtitle=StrongBodyLabel("Surrogate Modelling Method", self);self.subtitle.setObjectName("CardTitle")
        self.ContentWidget=QFrame(self);self.ContentWidget.setObjectName("MethodContentWidget")
        
        self.__initMethod()
        self.__initContentWidget()
        self.__initLayout()
        self.__initWidget()
        
        self.CurrentModel=0
        self.CurrentKernel=0
    def __initMethod(self):
        self.gp=GP_KernelCard()
        self.rbfn=RBFN_KernelCard()
        self.svm=SVM_KernelCard()
        self.mlp=MLP_KernelCard()
        self.pr=PR_KernelCard()
        self.kelm=KELM_KernelCard()
    def __initContentWidget(self):
        vBoxLayout=QVBoxLayout(self.ContentWidget)
        vBoxLayout.setContentsMargins(0,10,0,10)
        
        tempLayout=QHBoxLayout();tempLayout.addStretch(1);tempLayout.setAlignment(Qt.AlignLeft|Qt.AlignTop);tempLayout.setContentsMargins(10,0,0,0)
        self.methodCombobox=ComboBox();self.methodCombobox.setFixedWidth(380)
        self.ChooseTitle=BodyLabel("Surrogate Models:");self.ChooseTitle.setObjectName('ChooseTitle')
        tempLayout.addWidget(self.ChooseTitle,alignment=Qt.AlignmentFlag.AlignHCenter)
        tempLayout.addWidget(self.methodCombobox,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.methodCombobox.addItems(["Gaussian Process(GP)/Kriging(KRG)","Radial Basis Function(RBF)","Support Vector Machine(SVM)",
                              "Multilayer Perceptron(MLP)","Polynomial Regression(PR)", "Kernel Based Extreme Learning Machine (KELM)",
                              "Multi-Gene Genetic Programming(MGGP)","Ensemble Models(Heterogeneous or Homogeneous)"])
        self.stackWidget=QStackedWidget(self.ContentWidget);self.stackWidget.setObjectName("KernelWidget")
        self.stackWidget.setContentsMargins(0,0,0,0)
        self.stackWidget.addWidget(self.gp);self.stackWidget.addWidget(self.rbfn);self.stackWidget.addWidget(self.svm)
        self.stackWidget.addWidget(self.mlp);self.stackWidget.addWidget(self.pr);self.stackWidget.addWidget(self.kelm)
        self.stackWidget.addWidget(QFrame());self.stackWidget.addWidget(QFrame())
        tempLayout.addWidget(self.stackWidget,alignment=Qt.AlignTop)
        tempLayout.addStretch(1)
        vBoxLayout.addLayout(tempLayout);vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.methodCombobox.currentIndexChanged.connect(self.stackWidget.setCurrentIndex)
        self.methodCombobox.currentIndexChanged.connect(self.__SelectedModel)
        
        self.gp.BindIndex.connect(self.__SelectedSubMethod)
        self.rbfn.BindIndex.connect(self.__SelectedSubMethod)
        self.svm.BindIndex.connect(self.__SelectedSubMethod)
        self.mlp.BindIndex.connect(self.__SelectedSubMethod)
        self.pr.BindIndex.connect(self.__SelectedSubMethod)
        self.kelm.BindIndex.connect(self.__SelectedSubMethod)
        self.ModelList=[self.gp,self.rbfn,self.svm,self.mlp,self.pr,self.kelm]
    def __SelectedModel(self,i):
        self.CurrentModel=i
        self.BindStackWidget.emit(i)
        self.ModelList[i].kernelCombobox.setCurrentIndex(0)
        self.BindStackLabel.emit(0)
       
    def __SelectedSubMethod(self,i):
        self.BindStackKernel.emit(i)
        self.BindStackLabel.emit(i)
    def __initLayout(self):    
        self.vBoxLayout.addWidget(self.subtitle,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.vBoxLayout.addWidget(self.ContentWidget)
        
        self.bottomWidget=QFrame();self.bottomWidget.setObjectName('bottomWidget')
        bottomLayout=QHBoxLayout(self.bottomWidget);bottomLayout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.train=PushButton("Train Surrogate Models");self.verify=PushButton("Verify Surrogate Models");self.display=PushButton("Display Results")
        bottomLayout.addWidget(self.train);bottomLayout.addSpacing(100);bottomLayout.addWidget(self.verify);bottomLayout.addSpacing(100);bottomLayout.addWidget(self.display)
        self.vBoxLayout.addWidget(self.bottomWidget)
        
    def __initWidget(self):
        self.vBoxLayout.setContentsMargins(0,5,0,5)
    
class KernelSettingCard(StackWidget):
    ModelIndex=0
    MethodIndex=0
    K_index=[0,6,12,14,17,20]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Maximum)
        self.setObjectName("KernelSettingCard")
        #GP
        self.addWidget(gpKernelCard.RBFCard())
        self.addWidget(gpKernelCard.MaternCard())
        self.addWidget(gpKernelCard.RQCard())
        self.addWidget(gpKernelCard.ESSCard())
        self.addWidget(gpKernelCard.DotCard())
        self.addWidget(gpKernelCard.AdaptiveCard())
        #RBF
        self.addWidget(EmptyFrame())
        self.addWidget(EmptyFrame())
        self.addWidget(rbfnKernelCard.MultiCard())
        self.addWidget(rbfnKernelCard.ThinCard())
        self.addWidget(rbfnKernelCard.GasCard())
        self.addWidget(rbfnKernelCard.AdaptiveCard())
        #SVM
        self.addWidget(svmKernelCard.GasCard())
        self.addWidget(svmKernelCard.PolyCard())
        #MLP
        self.addWidget(mlpKernelCard.OneLayerCard())
        self.addWidget(mlpKernelCard.TwoLayerCard())
        self.addWidget(mlpKernelCard.ThreeLayerCard())
        #PR
        self.addWidget(EmptyFrame())
        self.addWidget(EmptyFrame())
        #KELM
        self.addWidget(EmptyFrame())
        self.addWidget(EmptyFrame())
        self.addWidget(EmptyFrame())
        
        
        self.adjustSize()
    def setBinder(self,binder):
        binder.BindStackWidget.connect(self.setModelIndex)
        binder.BindStackKernel.connect(self.setKernelIndex)
    def setModelIndex(self, index: int, needPopOut: bool = True, showNextWidgetDirectly: bool = False, duration: int = 500, easingCurve=QEasingCurve.OutQuad):
        self.ModelIndex=index
        self.setKernelIndex(0)
    def setKernelIndex(self,i):
        index=self.K_index[self.ModelIndex]+i
        if((self.Infos)[index]=="Empty"):
            self.setVisible(False)
        else:
            if(~self.isVisible()):
                self.setVisible(True)
            super().setCurrentIndex(index)
        
class EmptyFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent=None)
        self.setObjectName('Empty')
class ModelSettingCard(StackWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Maximum)
        self.setObjectName("ModelSettingCard")
        self.addWidget(GP_RightView())
        self.addWidget(RBF_RightView())
        self.addWidget(SVM_RightView())
        self.addWidget(MLP_RightView())
        self.addWidget(PR_RightView())
        self.setSizePolicy(QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Maximum)
    def setBinder(self,binder):
        binder.BindStackWidget.connect(self.setCurrentIndex)
#########################################################
class GP_RightView(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("GPRightView");self.setContentsMargins(5,0,5,5)
        self.rightLayout=QVBoxLayout(self);self.rightLayout.setAlignment(Qt.AlignTop)
        self.title=BodyLabel("Model Training Setting");self.title.setObjectName("CardTitle")
        self.rightLayout.addWidget(self.title,alignment=Qt.AlignHCenter)
        self.rightLayout.addSpacing(5)
        #Content
        tempLayout=QHBoxLayout();tempLayout.addStretch(1);tempLayout.addSpacing(12)
        labelTitle=BodyLabel("Alpha for diagonal:");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.alphaLineEdit=LineEdit(self);self.alphaLineEdit.setClearButtonEnabled(True)
        self.alphaLineEdit.setValidator(QDoubleValidator());self.alphaLineEdit.setText("0.001")
        self.alphaLineEdit.setMaximumWidth(350)
        tempLayout.addSpacing(5)
        tempLayout.addWidget(self.alphaLineEdit,alignment=Qt.AlignmentFlag.AlignLeft)
        tempLayout.addStretch(1)
        self.rightLayout.addLayout(tempLayout)
        
        tempLayout=QHBoxLayout();tempLayout.addStretch(1)
        labelTitle=BodyLabel("N_restarts_optimizer:");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.NRestartNumber=LineEdit(self)
        self.NRestartNumber.setClearButtonEnabled(True);self.NRestartNumber.setValidator(QIntValidator());self.alphaLineEdit.setMaximumWidth(350)
        self.NRestartNumber.setSizePolicy(QSizePolicy.Policy.Maximum,QSizePolicy.Policy.Maximum)
        self.NRestartNumber.setText("10")
        tempLayout.addSpacing(10);tempLayout.addWidget(self.NRestartNumber,alignment=Qt.AlignmentFlag.AlignLeft)
        tempLayout.addStretch(1)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addLayout(tempLayout)
        
        tempLayout=QHBoxLayout();tempLayout.addStretch(1);tempLayout.addSpacing(55)
        labelTitle=BodyLabel("Normalize_y:");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.switchNor_y=SwitchButton()
        tempLayout.addSpacing(5)
        tempLayout.addWidget(self.switchNor_y,alignment=Qt.AlignmentFlag.AlignLeft)
        tempLayout.addStretch(1)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addLayout(tempLayout)
        self.switchNor_y.checkedChanged.connect(self.onSwitchCheckedChanged)
        
    def onSwitchCheckedChanged(self, isChecked):
        if isChecked:
            self.switchNor_y.setText('On')
        else:
            self.switchNor_y.setText('Off')   
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen=QPen(QColor(0, 0, 0,38.25),1)
        painter.setPen(pen)
        h=self.title.height()+15
        w=self.width()
        painter.drawLine(QPointF(0,h),QPointF(w,h))
class RBF_RightView(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("RBFRightView");self.setContentsMargins(5,0,5,5)
        self.rightLayout=QVBoxLayout(self);self.rightLayout.setAlignment(Qt.AlignTop)
        self.title=BodyLabel("Model Training Setting");self.title.setObjectName("CardTitle")
        self.rightLayout.addWidget(self.title,alignment=Qt.AlignHCenter)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addStretch(1)
        #Content
        tempLayout=QHBoxLayout();tempLayout.addStretch(1)
        labelTitle=BodyLabel("Normalize_y:");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.switchNor_y=SwitchButton()
        tempLayout.addWidget(self.switchNor_y,alignment=Qt.AlignmentFlag.AlignLeft)
        tempLayout.addStretch(1)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addLayout(tempLayout)
        self.switchNor_y.checkedChanged.connect(self.onSwitchCheckedChanged)
        self.rightLayout.addStretch(1)
        # self.setFixedHeight(200)
        
    def onSwitchCheckedChanged(self, isChecked):
        if isChecked:
            self.switchNor_y.setText('On')
        else:
            self.switchNor_y.setText('Off')   
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen=QPen(QColor(0, 0, 0,38.25),1)
        painter.setPen(pen)
        h=self.title.height()+15
        w=self.width()
        painter.drawLine(QPointF(0,h),QPointF(w,h))
class SVM_RightView(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("SVMRightView");self.setContentsMargins(5,0,5,5)
        self.rightLayout=QVBoxLayout(self);self.rightLayout.setAlignment(Qt.AlignTop)
        self.title=BodyLabel("Model Training Setting");self.title.setObjectName("CardTitle")
        self.rightLayout.addWidget(self.title,alignment=Qt.AlignHCenter)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addStretch(1)
        #Content
        tempLayout=QHBoxLayout();tempLayout.addStretch(1)
        labelTitle=BodyLabel("N_Max_Iterations:");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.NIterations=LineEdit();self.NIterations.setValidator(QDoubleValidator())
        self.NIterations.setMaximumWidth(150)
        self.NIterations.setText('10000')
        tempLayout.addWidget(self.NIterations,alignment=Qt.AlignmentFlag.AlignLeft)
        tempLayout.addStretch(1)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addLayout(tempLayout)
        
        tempLayout=QHBoxLayout();tempLayout.addStretch(1)
        labelTitle=BodyLabel("Normalize_y:");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.switchNor_y=SwitchButton()
        tempLayout.addWidget(self.switchNor_y,alignment=Qt.AlignmentFlag.AlignLeft)
        tempLayout.addStretch(1)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addLayout(tempLayout)
        self.switchNor_y.checkedChanged.connect(self.onSwitchCheckedChanged)
        self.rightLayout.addStretch(1)
        # self.setFixedHeight(200)
        
    def onSwitchCheckedChanged(self, isChecked):
        if isChecked:
            self.switchNor_y.setText('On')
        else:
            self.switchNor_y.setText('Off')   
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen=QPen(QColor(0, 0, 0,38.25),1)
        painter.setPen(pen)
        h=self.title.height()+15
        w=self.width()
        painter.drawLine(QPointF(0,h),QPointF(w,h))
class MLP_RightView(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("MLPRightView");self.setContentsMargins(5,0,5,5)
        self.rightLayout=QVBoxLayout(self);self.rightLayout.setAlignment(Qt.AlignTop)
        self.title=BodyLabel("Model Training Setting");self.title.setObjectName("CardTitle")
        self.rightLayout.addWidget(self.title,alignment=Qt.AlignHCenter)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addStretch(1)
        #Content
        tempLayout=QHBoxLayout();tempLayout.addStretch(1)
        labelTitle=BodyLabel("Alpha:");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.NIterations=LineEdit();self.NIterations.setValidator(QDoubleValidator())
        self.NIterations.setMaximumWidth(75)
        self.NIterations.setText('0.0001')
        tempLayout.addWidget(self.NIterations,alignment=Qt.AlignmentFlag.AlignLeft)
        
        labelTitle=BodyLabel("Normalize_y:");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.switchNor_y=SwitchButton()
        tempLayout.addWidget(self.switchNor_y,alignment=Qt.AlignmentFlag.AlignLeft)
        
        
        tempLayout.addStretch(1)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addLayout(tempLayout)
        
        tempLayout=QHBoxLayout();tempLayout.addStretch(1)
        labelTitle=BodyLabel("Learning_rate");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.typeLearningRate=ComboBox();self.typeLearningRate.addItems(['constant', 'invscaling', 'adaptive'])
        self.typeLearningRate.setMinimumWidth(120)
        tempLayout.addWidget(self.typeLearningRate,alignment=Qt.AlignmentFlag.AlignLeft)
    
        labelTitle=BodyLabel("Solver");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.typeSolver=ComboBox();self.typeSolver.addItems(['lbfgs', 'sgd', 'adam'])
        self.typeSolver.setMinimumWidth(100)
        tempLayout.addWidget(self.typeSolver,alignment=Qt.AlignmentFlag.AlignLeft)
        tempLayout.addStretch(1)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addLayout(tempLayout)
        
        
        self.switchNor_y.checkedChanged.connect(self.onSwitchCheckedChanged)
        self.rightLayout.addStretch(1)
        # self.setFixedHeight(200)
        
    def onSwitchCheckedChanged(self, isChecked):
        if isChecked:
            self.switchNor_y.setText('On')
        else:
            self.switchNor_y.setText('Off')   
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen=QPen(QColor(0, 0, 0,38.25),1)
        painter.setPen(pen)
        h=self.title.height()+15
        w=self.width()
        painter.drawLine(QPointF(0,h),QPointF(w,h))
class PR_RightView(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("PRRightView");self.setContentsMargins(5,0,5,5)
        self.rightLayout=QVBoxLayout(self);self.rightLayout.setAlignment(Qt.AlignTop)
        self.title=BodyLabel("Model Training Setting");self.title.setObjectName("CardTitle")
        self.rightLayout.addWidget(self.title,alignment=Qt.AlignHCenter)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addStretch(1)
        #Content
        tempLayout=QHBoxLayout();tempLayout.addStretch(1)
        labelTitle=BodyLabel("Normalize_y:");labelTitle.setObjectName("ChooseTitle")
        tempLayout.addWidget(labelTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.switchNor_y=SwitchButton()
        tempLayout.addWidget(self.switchNor_y,alignment=Qt.AlignmentFlag.AlignLeft)
        tempLayout.addStretch(1)
        self.rightLayout.addSpacing(5)
        self.rightLayout.addLayout(tempLayout)
        self.switchNor_y.checkedChanged.connect(self.onSwitchCheckedChanged)
        self.rightLayout.addStretch(1)
        # self.setFixedHeight(200)
        
    def onSwitchCheckedChanged(self, isChecked):
        if isChecked:
            self.switchNor_y.setText('On')
        else:
            self.switchNor_y.setText('Off')   
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen=QPen(QColor(0, 0, 0,38.25),1)
        painter.setPen(pen)
        h=self.title.height()+15
        w=self.width()
        painter.drawLine(QPointF(0,h),QPointF(w,h))
#KernelCard Second Layer
class GP_KernelCard(QWidget):
    BindIndex=Signal(int)
    def __init__(self,parent=None):
        super().__init__(parent)
        kernelBoxLayout=QHBoxLayout(self)
        self.ChooseTitle=BodyLabel("The usage of kernel:");self.ChooseTitle.setObjectName("ChooseTitle")
        kernelBoxLayout.addWidget(self.ChooseTitle,alignment=Qt.AlignmentFlag.AlignLeft)
        self.kernelCombobox=ComboBox(self)
        self.kernelCombobox.addItems(["Squared-exponential kernel","Matern kernel","RationalQuadratic kernel","Periodic kernel","Dot Product Kernel","Automated Method"])
        self.kernelCombobox.setMinimumWidth(220)
        kernelBoxLayout.addWidget(self.kernelCombobox,alignment=Qt.AlignmentFlag.AlignLeft)
        kernelBoxLayout.addStretch(1)
        
        self.kernelCombobox.currentIndexChanged.connect(self.__sendCurrentIndex)
    def __sendCurrentIndex(self,i):
        self.BindIndex.emit(i)
class RBFN_KernelCard(QWidget):
    BindIndex=Signal(int)
    def __init__(self,parent=None):
        super().__init__(parent)
        kernelBoxLayout=QHBoxLayout(self);kernelBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ChooseTitle=BodyLabel("The usage of kernel:");self.ChooseTitle.setObjectName("ChooseTitle")
        kernelBoxLayout.addWidget(self.ChooseTitle,alignment=Qt.AlignLeft)
        self.kernelCombobox=ComboBox(self)
        self.kernelCombobox.addItems(["Cubic","Linear","Multi-Quadric","Thin Plate Spline","Gaussian","Adaptive Method"])
        self.kernelCombobox.setMinimumWidth(220)
        kernelBoxLayout.addWidget(self.kernelCombobox,alignment=Qt.AlignLeft)
        kernelBoxLayout.addStretch(1)
        
        self.kernelCombobox.currentIndexChanged.connect(self.__sendCurrentIndex)
    def __sendCurrentIndex(self,i):
        self.BindIndex.emit(i)
class SVM_KernelCard(QWidget):
    BindIndex=Signal(int)
    def __init__(self,parent=None):
        super().__init__(parent)
        kernelBoxLayout=QHBoxLayout(self);kernelBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ChooseTitle=BodyLabel("The usage of kernel:");self.ChooseTitle.setObjectName("ChooseTitle")
        kernelBoxLayout.addWidget(self.ChooseTitle,alignment=Qt.AlignLeft)
        self.kernelCombobox=ComboBox(self)
        self.kernelCombobox.addItems(["Gaussian","Polynomial"])
        self.kernelCombobox.setMinimumWidth(220)
        kernelBoxLayout.addWidget(self.kernelCombobox,alignment=Qt.AlignLeft)
        kernelBoxLayout.addStretch(1)
        
        self.kernelCombobox.currentIndexChanged.connect(self.__sendCurrentIndex)
    def __sendCurrentIndex(self,i):
        self.BindIndex.emit(i)
class MLP_KernelCard(QWidget):
    BindIndex=Signal(int)
    def __init__(self,parent=None):
        super().__init__(parent)
        kernelBoxLayout=QHBoxLayout(self);kernelBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ChooseTitle=BodyLabel("The numbers of hidden layers:");self.ChooseTitle.setObjectName("ChooseTitle")
        kernelBoxLayout.addWidget(self.ChooseTitle,alignment=Qt.AlignLeft)
        self.kernelCombobox=ComboBox(self)
        self.kernelCombobox.addItems(["1","2","3"])
        self.kernelCombobox.setMinimumWidth(120)
        kernelBoxLayout.addWidget(self.kernelCombobox,alignment=Qt.AlignLeft)
        kernelBoxLayout.addStretch(1)
        
        self.kernelCombobox.currentIndexChanged.connect(self.__sendCurrentIndex)
    def __sendCurrentIndex(self,i):
        self.BindIndex.emit(i)
        
class PR_KernelCard(QWidget):
    BindIndex=Signal(int)
    def __init__(self,parent=None):
        super().__init__(parent)
        kernelBoxLayout=QHBoxLayout(self);kernelBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ChooseTitle=BodyLabel("The degree:");self.ChooseTitle.setObjectName("ChooseTitle")
        kernelBoxLayout.addWidget(self.ChooseTitle,alignment=Qt.AlignLeft)
        self.kernelCombobox=ComboBox(self)
        self.kernelCombobox.addItems(["2","3"])
        self.kernelCombobox.setMinimumWidth(120)
        kernelBoxLayout.addWidget(self.kernelCombobox,alignment=Qt.AlignLeft)
        kernelBoxLayout.addStretch(1)
        
        self.kernelCombobox.currentIndexChanged.connect(self.__sendCurrentIndex)
    def __sendCurrentIndex(self,i):
        self.BindIndex.emit(i)

class KELM_KernelCard(QWidget):
    BindIndex=Signal(int)
    def __init__(self,parent=None):
        super().__init__(parent)
        kernelBoxLayout=QHBoxLayout(self);kernelBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.ChooseTitle=BodyLabel("The usage of kernel:");self.ChooseTitle.setObjectName("ChooseTitle")
        kernelBoxLayout.addWidget(self.ChooseTitle,alignment=Qt.AlignLeft)
        self.kernelCombobox=ComboBox(self)
        self.kernelCombobox.addItems(["Gaussian","Linear","Polynomial"])
        self.kernelCombobox.setMinimumWidth(220)
        kernelBoxLayout.addWidget(self.kernelCombobox,alignment=Qt.AlignLeft)
        kernelBoxLayout.addStretch(1)
        
        self.kernelCombobox.currentIndexChanged.connect(self.__sendCurrentIndex)
    def __sendCurrentIndex(self,i):
        self.BindIndex.emit(i)
################################################
        
        
        
        
        
        
        
        
               
        
        
        
        
        



            
            
            
            
            
            
            
            
        
           
           
           
        
        
        
        
        