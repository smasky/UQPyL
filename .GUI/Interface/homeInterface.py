from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QIcon,QPainter, QBrush, QPainterPath
from PySide6.QtWidgets import QApplication, QLabel

from PySide6.QtCore import Qt, Signal, QEasingCurve, QUrl, QSize,QTimer,QRectF,QPointF
# from qframelesswindow import FramelessWindow, TitleBar, StandardTitleBar
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QPushButton,QHBoxLayout,QLabel,QSizePolicy
from qframelesswindow import FramelessWindow,StandardTitleBar
from PySide6.QtGui import QColor, QPixmap, QIcon,QColor,QPalette, QLinearGradient,QGradient,QBrush

from qfluentwidgets import (NavigationAvatarWidget, NavigationItemPosition, MessageBox, FluentWindow,
                            SplashScreen,ScrollArea)
from qfluentwidgets import FluentIcon
from .Compoent.linkView import LinkCardView
from .Compoent import BannerWidget
class HomeInterface(ScrollArea):
    ''' HomePage '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.banner = BannerWidget(self)
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self.view)
        self.linkCardView = LinkCardView(self.banner)
        
        self.__initBanner()
        self.__initWidget() 
        self.loadSamples() #TODO
    def __initBanner(self):
        self.banner.setTitle("UQ-PyL")
        self.banner.setPixmap("./picture/header.png")
        self.__iniLinkView()
        self.banner.addWidget(self.linkCardView)
        self.banner.vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
    def __initWidget(self):
        self.view.setObjectName('view')
        self.setObjectName('homeInterface')
        
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.vBoxLayout.setContentsMargins(0, 0, 0, 36)
        self.vBoxLayout.setSpacing(40)
        self.vBoxLayout.addWidget(self.banner)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        
        #set Qss
        with open("./qss//home_interface.qss") as f:
            self.setStyleSheet(f.read())
    def loadSamples(self):
        pass
    def __iniLinkView(self):
        self.linkCardView.view.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Maximum)
        self.linkCardView.hBoxLayout.setContentsMargins(28,20,0,0)
        self.linkCardView.addCard(
            './picture/ICON-small.png',
            self.tr('Quick Start'),
            self.tr('An overview of UQ-PyL and quickly start for your use.'),
            "http://www.uq-pyl.com/"
        )

        self.linkCardView.addCard(
            FluentIcon.GITHUB,
            self.tr('Code Repo'),
            self.tr(
                'The latest version applications and shell controls for usage.'),
            "http://www.uq-pyl.com/"
        )

        self.linkCardView.addCard(
            FluentIcon.CODE,
            self.tr('Project Examples'),
            self.tr(
                'Find already project examples that demonstrate features.'),
            "http://www.uq-pyl.com/"
        )

        self.linkCardView.addCard(
            FluentIcon.FEEDBACK,
            self.tr('Join Us'),
            self.tr('Help us improve UQ-PyL and contribute your efforts for it.'),
           "http://www.uq-pyl.com/"
        )