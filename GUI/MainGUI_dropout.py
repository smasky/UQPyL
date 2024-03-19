import sys
import os
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QPushButton,QHBoxLayout,QLabel, QSplitter,QSizePolicy
from qframelesswindow import FramelessWindow,StandardTitleBar
from PySide6.QtGui import QColor, QPixmap, QIcon,QColor,QPalette
os.chdir("./GUI")
class Window(FramelessWindow):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        #Basic Setting
        self.setTitleBar(StandardTitleBar(self))
        self.titleBar.setObjectName("TitleBar")
        self.setObjectName("MainWidget")
        self.titleBar.setIcon("./picture/title_icon.png")
        self.titleBar.setTitle("UQ-PyL")
        self.titleBar.titleLabel.setStyleSheet("color:#0d1b2a;font-weight:bold;font-family:'Arial';")
        self.titleBar.raise_()
        self.resize(1200,800)
        #Self.setStyleSheet("background-color:red")
        
        #Framework Design
        MainLayout=QHBoxLayout(self) #Horizontal
        MainLayout.setContentsMargins(0,32,0,0)
        
        #Control Definition
        Problems_Switch=QPushButton("Problem Definition");Problems_Switch.setObjectName("Switch")
        DOE_Switch=QPushButton("Design of Experiment");DOE_Switch.setObjectName("Switch")
        Uncertainty_Switch=QPushButton("Uncertainty Analysis");Uncertainty_Switch.setObjectName("Switch")
        Surrogate_Switch=QPushButton("Surrogate Modelling");Surrogate_Switch.setObjectName("Switch")
        Optimization=QPushButton("Optimization");Optimization.setObjectName("Switch")
        # Splitter1=QWidget();Splitter1.setObjectName("Splitter")
        # Splitter1.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Maximum); Splitter1.setMinimumHeight(1)
        
        with open("./qss/MainGUI.qss") as f:
            self.setStyleSheet(f.read())
        
        #Self.setStyleSheet("background-color:red")
        Left_Main_Widget=QWidget();Left_Main_Widget.setObjectName("LeftMain")
        Right_Main_Widget=QWidget();Right_Main_Widget.setObjectName("RightMain")
        Right_Main_Widget.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding);
        
        Left_First_Layout=QVBoxLayout(Left_Main_Widget);Left_Main_Widget.setLayout(Left_First_Layout)
        Left_First_Layout.addStretch(1)
        Left_First_Layout.setSpacing(40)
        Left_First_Layout.setContentsMargins(0,0,0,0)
        Left_First_Layout.addWidget(Problems_Switch)
        Left_First_Layout.addWidget(DOE_Switch)
        Left_First_Layout.addWidget(Uncertainty_Switch)
        Left_First_Layout.addWidget(Surrogate_Switch)
        Left_First_Layout.addWidget(Optimization)
        Left_First_Layout.addStretch(5)
        
        MainLayout.addWidget(Left_Main_Widget)
        MainLayout.addWidget(Right_Main_Widget)
        MainLayout.setSpacing(0)
        
        
        

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = Window()
    demo.show()
    app.exec()