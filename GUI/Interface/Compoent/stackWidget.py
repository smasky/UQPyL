from typing import Optional
from PySide6.QtWidgets import QStackedWidget

class StackWidget(QStackedWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.Infos=[]
    def addWidget(self, widget):
        super().addWidget(widget)
        self.Infos.append(widget.objectName())