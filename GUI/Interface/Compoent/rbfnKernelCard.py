from PySide6.QtCore import Qt,QPointF
from PySide6.QtGui import QDoubleValidator,QPainter,QPen,QColor
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QFrame,QSizePolicy
from qfluentwidgets import (BodyLabel,LineEdit,SwitchButton,PillPushButton,FlowLayout,SpinBox)
#Abandon
# class CubicCard(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.vBoxLayout=QVBoxLayout(self)
#         self.vBoxLayout.setContentsMargins(0,0,0,0)
#         hBoxLayout=QHBoxLayout();hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
#         hBoxLayout.addWidget(BodyLabel("Normalize_y:"))
#         self.switchNor_y=SwitchButton("On")
#         hBoxLayout.addWidget(self.switchNor_y)
        
#         self.vBoxLayout.addStretch(1)
#         self.vBoxLayout.addLayout(hBoxLayout)
#         self.vBoxLayout.addStretch(1)
# class LinearCard(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.vBoxLayout=QVBoxLayout(self)
#         self.vBoxLayout.setContentsMargins(0,0,0,0)
#         hBoxLayout=QHBoxLayout();hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
#         hBoxLayout.addWidget(BodyLabel("Normalize_y:"))
#         self.switchNor_y=SwitchButton("On")
#         hBoxLayout.addWidget(self.switchNor_y)
        
#         self.vBoxLayout.addStretch(1)
#         self.vBoxLayout.addLayout(hBoxLayout)
#         self.vBoxLayout.addStretch(1)
class MultiCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("KernelView");self.setContentsMargins(5,0,5,5)
        self.leftLayout=QVBoxLayout(self);self.leftLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.title=BodyLabel("Kernel Parameter Card");self.title.setObjectName('CardTitle')
        self.leftLayout.addWidget(self.title,alignment=Qt.AlignmentFlag.AlignHCenter)
        
        hBoxLayout=QHBoxLayout()
        hBoxLayout.addSpacing(20);hBoxLayout.addStretch(1)
        title=BodyLabel("Gamma:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.lengthLineEdit=LineEdit(self)
        self.lengthLineEdit.setValidator(QDoubleValidator())
        self.lengthLineEdit.setMaximumWidth(100);self.lengthLineEdit.setText("1.00")
        hBoxLayout.addWidget(self.lengthLineEdit,alignment=Qt.AlignmentFlag.AlignHCenter)
        hBoxLayout.addStretch(1)
        self.leftLayout.addSpacing(5)
        self.leftLayout.addLayout(hBoxLayout)
        
        hBoxLayout=QHBoxLayout();hBoxLayout.addStretch(1)
        title=BodyLabel("Optimize Gamma:");title.setObjectName("ChooseTitle")
        hBoxLayout.addWidget(title)
        self.isOptimize=SwitchButton()
        self.isOptimize.toggleChecked()
        hBoxLayout.addWidget(self.isOptimize)
        hBoxLayout.addStretch(1)
        self.leftLayout.addSpacing(5)
        self.leftLayout.addLayout(hBoxLayout)
        
        hBoxLayout=QHBoxLayout();hBoxLayout.addStretch(1)
        title=BodyLabel("Gamma LB:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title)
        self.lowLineEdit=LineEdit(self)
        self.lowLineEdit.setValidator(QDoubleValidator())
        self.lowLineEdit.setMaximumWidth(70)
        self.lowLineEdit.setText('0')
        hBoxLayout.addWidget(self.lowLineEdit)
        title=BodyLabel("Gamma UB:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title)
        self.upLineEdit=LineEdit(self)
        self.upLineEdit.setValidator(QDoubleValidator())
        self.upLineEdit.setMaximumWidth(70)
        self.upLineEdit.setText('100')
        hBoxLayout.addWidget(self.upLineEdit)
        hBoxLayout.addStretch(1)
        self.leftLayout.addSpacing(5)
        self.leftLayout.addLayout(hBoxLayout)
        
        self.isOptimize.checkedChanged.connect(self.__onSwitchOPChanged)
    def __onSwitchOPChanged(self,isChecked):
        if isChecked:
            self.upLineEdit.setEnabled(True)
            self.lowLineEdit.setEnabled(True)
        else:
            self.lowLineEdit.setEnabled(False)
            self.upLineEdit.setEnabled(False)
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen=QPen(QColor(0, 0, 0,38.25),1)
        painter.setPen(pen)
        h=self.title.height()+15
        w=self.width()
        painter.drawLine(QPointF(0,h),QPointF(w,h))
        
class GasCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("KernelView");self.setContentsMargins(5,0,5,5)
        self.leftLayout=QVBoxLayout(self);self.leftLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.title=BodyLabel("Kernel Parameter Card");self.title.setObjectName('CardTitle')
        self.leftLayout.addWidget(self.title,alignment=Qt.AlignmentFlag.AlignHCenter)
        
        hBoxLayout=QHBoxLayout()
        hBoxLayout.addSpacing(20);hBoxLayout.addStretch(1)
        title=BodyLabel("Gamma:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.lengthLineEdit=LineEdit(self)
        self.lengthLineEdit.setClearButtonEnabled(True)
        self.lengthLineEdit.setValidator(QDoubleValidator())
        self.lengthLineEdit.setMaximumWidth(100);self.lengthLineEdit.setText("1.00")
        hBoxLayout.addWidget(self.lengthLineEdit,alignment=Qt.AlignmentFlag.AlignHCenter)
        hBoxLayout.addStretch(1)
        self.leftLayout.addSpacing(5)
        self.leftLayout.addLayout(hBoxLayout)
        
        hBoxLayout=QHBoxLayout();hBoxLayout.addStretch(1)
        title=BodyLabel("Optimize Gamma:");title.setObjectName("ChooseTitle")
        hBoxLayout.addWidget(title)
        self.isOptimize=SwitchButton()
        self.isOptimize.toggleChecked()
        hBoxLayout.addWidget(self.isOptimize)
        hBoxLayout.addStretch(1)
        self.leftLayout.addSpacing(5)
        self.leftLayout.addLayout(hBoxLayout)
        
        hBoxLayout=QHBoxLayout();hBoxLayout.addStretch(1)
        title=BodyLabel("Gamma LB:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title)
        self.lowLineEdit=LineEdit(self)
        self.lowLineEdit.setValidator(QDoubleValidator())
        self.lowLineEdit.setMaximumWidth(70)
        self.lowLineEdit.setText('0')
        hBoxLayout.addWidget(self.lowLineEdit)
        title=BodyLabel("Gamma UB:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title)
        self.upLineEdit=LineEdit(self)
        self.upLineEdit.setValidator(QDoubleValidator())
        self.upLineEdit.setMaximumWidth(70)
        self.upLineEdit.setText('100')
        hBoxLayout.addWidget(self.upLineEdit)
        hBoxLayout.addStretch(1)
        self.leftLayout.addSpacing(5)
        self.leftLayout.addLayout(hBoxLayout)
        
        self.isOptimize.checkedChanged.connect(self.__onSwitchOPChanged)
    def __onSwitchOPChanged(self,isChecked):
        if isChecked:
            self.upLineEdit.setEnabled(True)
            self.lowLineEdit.setEnabled(True)
        else:
            self.lowLineEdit.setEnabled(False)
            self.upLineEdit.setEnabled(False)
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen=QPen(QColor(0, 0, 0,38.25),1)
        painter.setPen(pen)
        h=self.title.height()+15
        w=self.width()
        painter.drawLine(QPointF(0,h),QPointF(w,h))
class ThinCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("KernelView");self.setContentsMargins(5,0,5,5)
        self.leftLayout=QVBoxLayout(self);self.leftLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.title=BodyLabel("Kernel Parameter Card");self.title.setObjectName('CardTitle')
        self.leftLayout.addWidget(self.title,alignment=Qt.AlignmentFlag.AlignHCenter)
        
        hBoxLayout=QHBoxLayout()
        hBoxLayout.addSpacing(20);hBoxLayout.addStretch(1)
        title=BodyLabel("Gamma:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.lengthLineEdit=LineEdit(self)
        self.lengthLineEdit.setClearButtonEnabled(True)
        self.lengthLineEdit.setValidator(QDoubleValidator())
        self.lengthLineEdit.setMaximumWidth(100);self.lengthLineEdit.setText("1.00")
        hBoxLayout.addWidget(self.lengthLineEdit,alignment=Qt.AlignmentFlag.AlignHCenter)
        hBoxLayout.addStretch(1)
        self.leftLayout.addSpacing(5)
        self.leftLayout.addLayout(hBoxLayout)
        
        hBoxLayout=QHBoxLayout();hBoxLayout.addStretch(1)
        title=BodyLabel("Optimize Gamma:");title.setObjectName("ChooseTitle")
        hBoxLayout.addWidget(title)
        self.isOptimize=SwitchButton()
        self.isOptimize.toggleChecked()
        hBoxLayout.addWidget(self.isOptimize)
        hBoxLayout.addStretch(1)
        self.leftLayout.addSpacing(5)
        self.leftLayout.addLayout(hBoxLayout)
        
        hBoxLayout=QHBoxLayout();hBoxLayout.addStretch(1)
        title=BodyLabel("Gamma LB:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title)
        self.lowLineEdit=LineEdit(self)
        self.lowLineEdit.setValidator(QDoubleValidator())
        self.lowLineEdit.setMaximumWidth(70)
        self.lowLineEdit.setText('0')
        hBoxLayout.addWidget(self.lowLineEdit)
        title=BodyLabel("Gamma UB:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title)
        self.upLineEdit=LineEdit(self)
        self.upLineEdit.setValidator(QDoubleValidator())
        self.upLineEdit.setMaximumWidth(70)
        self.upLineEdit.setText('100')
        hBoxLayout.addWidget(self.upLineEdit)
        hBoxLayout.addStretch(1)
        self.leftLayout.addSpacing(5)
        self.leftLayout.addLayout(hBoxLayout)
        
        self.isOptimize.checkedChanged.connect(self.__onSwitchOPChanged)
    def __onSwitchOPChanged(self,isChecked):
        if isChecked:
            self.upLineEdit.setEnabled(True)
            self.lowLineEdit.setEnabled(True)
        else:
            self.lowLineEdit.setEnabled(False)
            self.upLineEdit.setEnabled(False)
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen=QPen(QColor(0, 0, 0,38.25),1)
        painter.setPen(pen)
        h=self.title.height()+15
        w=self.width()
        painter.drawLine(QPointF(0,h),QPointF(w,h))
class AdaptiveCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("KernelView")
        self.leftLayout=QVBoxLayout(self);self.leftLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.leftLayout.addSpacing(5)
        self.title=BodyLabel("Automated Technique Card");self.title.setObjectName('CardTitle')
        self.leftLayout.addWidget(self.title,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.leftLayout.setContentsMargins(0,0,0,0)
        self.flowWidget=QFrame();self.flowWidget.setStyleSheet("border-bottom:1px solid rgba(0, 0, 0, 0.15)")
        self.flowLayout=FlowLayout(self.flowWidget);self.leftLayout.addWidget(self.flowWidget)
        self.flowWidget.setStyleSheet("border-bottom: 1px solid rgba(0, 0, 0, 0.15);")
        self.flowLayout.setContentsMargins(5,10,5,0)
        texts=["Cubic Kernel","Linear Kernel","Multi-Quadric Kernel","Thin Plate Spline Kernel","Gaussian Kernel"]
        self.__generateFlowWidget(texts)
        
        
        hBoxLayout=QHBoxLayout();hBoxLayout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        hBoxLayout.addSpacing(5)
        title=BodyLabel("K of the K-fold Cross Validation method:");title.setObjectName('ChooseTitle')
        hBoxLayout.addWidget(title)
        self.KLineEdit=SpinBox()
        self.KLineEdit.setRange(1,20)
        self.KLineEdit.setMaximumWidth(350)
        hBoxLayout.addWidget(self.KLineEdit)
        self.leftLayout.addLayout(hBoxLayout)
        self.leftLayout.addStretch(1)
        
    def __generateFlowWidget(self,texts):
        for i in reversed(range(self.flowLayout.count())): 
            self.flowLayout.itemAt(i).widget().setParent(None)
        for text in texts:
            self.flowLayout.addWidget(PillPushButton(text))  
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        pen=QPen(QColor(0, 0, 0,38.25),1)
        painter.setPen(pen)
        h=self.title.height()+15
        w=self.width()
        painter.drawLine(QPointF(0,h),QPointF(w,h))