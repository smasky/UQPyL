from PySide6.QtWidgets import QWidget, QVBoxLayout,QLabel,QSizePolicy,QFrame, QHBoxLayout,QPushButton
from PySide6.QtGui import QPixmap, QPainter,QLinearGradient, QColor, QGradient, QPainterPath,QBrush
from PySide6.QtCore import Qt,QPointF, QRectF, QSize,Signal
from qfluentwidgets import IconWidget,FluentStyleSheet, PushButton, CaptionLabel, SubtitleLabel,StrongBodyLabel



class SettingCard(QFrame):
    """ Setting card """

    def __init__(self, icon, title, content=None, parent=None):
        """
        Parameters
        ----------
        icon: str | QIcon | FluentIconBase
            the icon to be drawn

        title: str
            the title of card

        content: str
            the content of card

        parent: QWidget
            parent widget
        """
        super().__init__(parent=parent)
        self.iconLabel = IconWidget(icon, self)
        self.titleLabel = QLabel(title, self)
        self.contentLabel = QLabel(content or '', self)
        self.hBoxLayout = QHBoxLayout(self)
        self.vBoxLayout = QVBoxLayout()

        if not content:
            self.contentLabel.hide()

        self.setFixedHeight(50 if content else 30)
        self.iconLabel.setFixedSize(20, 20)

        # initialize layout
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(16, 0, 0, 0)
        self.hBoxLayout.setAlignment(Qt.AlignVCenter)
        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setAlignment(Qt.AlignVCenter)

        self.hBoxLayout.addWidget(self.iconLabel, 0, Qt.AlignLeft)
        self.hBoxLayout.addSpacing(8)

        self.hBoxLayout.addLayout(self.vBoxLayout)
        self.vBoxLayout.addWidget(self.titleLabel, 0, Qt.AlignLeft)
        self.vBoxLayout.addWidget(self.contentLabel, 0, Qt.AlignLeft)

        self.hBoxLayout.addSpacing(8)
        self.hBoxLayout.addStretch(1)
        
        self.titleLabel.setObjectName('titleLabel')
        self.contentLabel.setObjectName('contentLabel')
        # FluentStyleSheet.SETTING_CARD.apply(self)

    def setTitle(self, title: str):
        """ set the title of card """
        self.titleLabel.setText(title)

    def setContent(self, content: str):
        """ set the content of card """
        self.contentLabel.setText(content)
        self.contentLabel.setVisible(bool(content))

    def setValue(self, value):
        """ set the value of config item """
        pass

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)

        if False:
            painter.setBrush(QColor(255, 255, 255, 13))
            painter.setPen(QColor(0, 0, 0, 50))
        else:
            painter.setBrush(QColor(255, 255, 255, 170))
            painter.setPen(QColor(0, 0, 0, 19))

        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 10, 10)
        
        
class PushSettingCard(SettingCard):
    """ Setting card with a push button """

    clicked = Signal()

    def __init__(self, text, icon, title, content=None, ObjectiveName="",parent=None):
        """
        Parameters
        ----------
        text: str
            the text of push button

        icon: str | QIcon | FluentIconBase
            the icon to be drawn

        title: str
            the title of card

        content: str
            the content of card

        parent: QWidget
            parent widget
        """
        super().__init__(icon, title, content, parent)
        self.button = PushButton(text, self)
        self.hBoxLayout.addWidget(self.button, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
        self.button.clicked.connect(self.clicked)
        