# coding:utf-8
import sys
import os
os.chdir("./GUI")
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QIcon,QPainter, QBrush, QPainterPath
from PySide6.QtWidgets import QApplication, QLabel

from PySide6.QtCore import Qt, Signal, QEasingCurve, QUrl, QSize,QTimer,QRectF
# from qframelesswindow import FramelessWindow, TitleBar, StandardTitleBar
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QPushButton,QHBoxLayout,QLabel
from qframelesswindow import FramelessWindow,StandardTitleBar
from PySide6.QtGui import QColor, QPixmap, QIcon,QColor,QPalette

from qfluentwidgets import (NavigationAvatarWidget, NavigationItemPosition, MessageBox, FluentWindow,
                            SplashScreen,ScrollArea,NavigationDisplayMode)
from qfluentwidgets import FluentIcon as FIF


from Interface import HomeInterface, ProblemInterface,DOEInterface,surrogateInterface

class Window(FluentWindow):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.initWindow()
        
        self.homeInterface = HomeInterface(self) #TODO
        self.homeInterface.setObjectName("Home")
        self.ProblemInterface = ProblemInterface(self) #TODO
        self.ProblemInterface.setObjectName("Problem Definition")
        self.DOEInterface = DOEInterface(self) #TODO
        self.DOEInterface.setObjectName("Design of Experiment")
        self.SurrogateInterface = surrogateInterface(self) #TODO
        self.SurrogateInterface.setObjectName("Surrogate Modelling")
        self.UncertaintyInterface = QWidget(self) #TODO
        self.UncertaintyInterface.setObjectName("Uncertainty Analysis")
        self.SensitivityInterface = QWidget(self) #TODO
        self.SensitivityInterface.setObjectName("Sensitivity Analysis")
        self.OptimizationInterface = QWidget(self) #TODO
        self.OptimizationInterface.setObjectName("Optimization")
        self.SettingInterface = QWidget(self) #TODO
        self.SettingInterface.setObjectName("Setting")
        
        self.initNavigation()
        
        self.splashScreen.finish()
        Time=QTimer(self)
        self.navigationInterface.panel.toggle()
        
        QTimer.singleShot(500,self.navigationInterface.panel.toggle)
        
        # self.navigationInterface.panel.setReturnButtonVisible(showReturnButton)
        # self.navigationInterface.displayModeChanged.emit(1)
    def initNavigation(self):
        ###Top Navigation
        self.addSubInterface(self.homeInterface, FIF.HOME, self.tr('Home')) #CALORIES
        self.navigationInterface.addSeparator()
        
        ####Scroll Navigation
        pos=NavigationItemPosition.SCROLL
        self.addSubInterface(self.ProblemInterface, FIF.CALORIES, self.tr('Problem Definition'),pos)
        self.addSubInterface(self.DOEInterface,FIF.ROBOT,self.tr('Design of Experiment'),pos)
        self.addSubInterface(self.SurrogateInterface,FIF.ALBUM,self.tr('Surrogate Modelling'),pos)
        self.navigationInterface.addSeparator(position=pos)
        self.addSubInterface(self.UncertaintyInterface,FIF.GAME,self.tr('Uncertainty Analysis'),pos)
        self.addSubInterface(self.SensitivityInterface,FIF.IOT,self.tr('Sensitivity Analysis'),pos)
        self.addSubInterface(self.OptimizationInterface,FIF.BRUSH,self.tr('Optimization'),pos)
        self.navigationInterface.panel.scrollLayout.setSpacing(10)
        
        ####Bottom Navigation
        self.addSubInterface(
            self.SettingInterface, FIF.SETTING, self.tr('Settings'), NavigationItemPosition.BOTTOM)
        
    def initWindow(self):
        self.resize(1024, 780)
        self.setMinimumWidth(760)
        self.setWindowIcon(QIcon('./picture/title_icon.png'))
        self.setWindowTitle('UQ-PyL2.0')

        self.setMicaEffectEnabled(False)
        # # create splash screen
        self.splashScreen = SplashScreen(QIcon('./picture/splash.png'), self)
        self.splashScreen.setIconSize(QSize(500, 500))
        self.splashScreen.raise_()

        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w//2 - self.width()//2, h//2 - self.height()//2)
        self.show()
        QApplication.processEvents()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = Window()
    demo.show()
    app.exec()
    