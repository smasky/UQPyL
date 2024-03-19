from PySide6.QtWidgets import QWidget, QVBoxLayout,QLabel,QSizePolicy
from PySide6.QtGui import QPixmap, QPainter,QLinearGradient, QColor, QGradient, QPainterPath,QBrush
from PySide6.QtCore import Qt,QPointF, QRectF, QSize



class BannerWidget(QWidget):
    """Banner widget of HomePage"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.vBoxLayout = QVBoxLayout(self)
        self.galleryLabel = QLabel(self)
        self.banner = QPixmap('')
        
        self.galleryLabel.setObjectName('galleryLabel')
        self.vBoxLayout.addWidget(self.galleryLabel)
        # self.vBoxLayout.addWidget(SegmentedInterface(self))
        self.vBoxLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
    def addWidget(self,widget):
        self.vBoxLayout.addWidget(widget)
    def setTitle(self,text):
        self.galleryLabel.setText(text)
    def setPixmap(self,path):
        self.banner=QPixmap(path)
    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        #
        line=QLinearGradient(QPointF(0,0),QPointF(0,336))
        line.setColorAt(0, QColor("#ffced8e4"))
        # line.setColorAt(0.5, QColor(206, 216, 228,255))
        line.setColorAt(1, QColor("#00ffffff"))
        # line.setSpread(QGradient.Spread.RepeatSpread)
        painter.setBrush(line)
        
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        w, h = self.width(), 336
        path.addRoundedRect(QRectF(0, 0, w, h), 10, 10)
        path.addRect(QRectF(0, h-50, 50, 50))
        path.addRect(QRectF(w-50, 0, 50, 50))
        path.addRect(QRectF(w-50, h-50, 50, 50))
        path = path.simplified()
        painter.fillPath(path, QBrush(line))
        # draw banner image
        pixmap = self.banner.scaled(
            QSize(self.width(),336), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        path.addRect(QRectF(0, 336, w, 336 ))
        painter.drawPixmap(0,0,self.width(),336,pixmap)