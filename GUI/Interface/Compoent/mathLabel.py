from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter,QTextDocument
from PySide6.QtWidgets import QApplication, QLabel, QWidget

class MathLabel(QLabel):
    def __init__(self,text):
        super().__init__(text)
        self.setFixedHeight(50)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        document = QTextDocument(self)
        document.setHtml("<p>{}</p>".format(self.text()))
        document.drawContents(painter)
    