# coding: utf-8
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QCheckBox, QStyle, QStyleOptionButton, QWidget

from ...common.icon import FluentIconBase, Theme, getIconColor
from ...common.style_sheet import FluentStyleSheet
from ...common.overload import singledispatchmethod


class CheckBoxIcon(FluentIconBase, Enum):
    """ CheckBoxIcon """

    ACCEPT = "Accept"
    PARTIAL_ACCEPT = "PartialAccept"

    def path(self, theme=Theme.AUTO):
        c = getIconColor(theme, reverse=True)
        return f':/qfluentwidgets/images/check_box/{self.value}_{c}.svg'



class CheckBox(QCheckBox):
    """ Check box """

    @singledispatchmethod
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        FluentStyleSheet.CHECK_BOX.apply(self)

    @__init__.register
    def _(self, text: str, parent: QWidget = None):
        self.__init__(parent)
        self.setText(text)

    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)

        if not self.isEnabled():
            painter.setOpacity(0.8)

        # get the rect of indicator
        opt = QStyleOptionButton()
        opt.initFrom(self)
        rect = self.style().subElementRect(QStyle.SE_CheckBoxIndicator, opt, self)

        # draw indicator
        if self.checkState() == Qt.Checked:
            CheckBoxIcon.ACCEPT.render(painter, rect)
        elif self.checkState() == Qt.PartiallyChecked:
            CheckBoxIcon.PARTIAL_ACCEPT.render(painter, rect)
