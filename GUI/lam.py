import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from sympy import symbols, Eq, init_printing, latex

# 创建GUI窗口
app = QApplication(sys.argv)
window = QWidget()
layout = QVBoxLayout()
window.setLayout(layout)

# 使用SymPy生成LaTeX公式
init_printing(use_latex=True)
x, y = symbols('x y')
equation = Eq(x**2 + y**2, 1)
equation_latex = latex(equation)
label = QLabel(equation_latex)
label.setAlignment(Qt.AlignCenter)
layout.addWidget(label)

# 运行GUI应用程序
window.show()
sys.exit(app.exec_())