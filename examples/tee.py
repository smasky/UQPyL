import numpy as np

class MatrixWrapper:
    def __init__(self, matrix):
        self.matrix = np.array(matrix)
    
    # 实现加法操作
    def __add__(self, other):
        if isinstance(other, (MatrixWrapper, np.ndarray)):
            return MatrixWrapper(self.matrix + other.matrix if isinstance(other, MatrixWrapper) else self.matrix + other)
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")

    # 实现减法操作
    def __sub__(self, other):
        if isinstance(other, (MatrixWrapper, np.ndarray)):
            return MatrixWrapper(self.matrix - other.matrix if isinstance(other, MatrixWrapper) else self.matrix - other)
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")
    
    # 反向加法（例如：np_array + MatrixWrapper）
    def __radd__(self, other):
        return self.__add__(other)
    
    # 反向减法
    def __rsub__(self, other):
        if isinstance(other, np.ndarray):
            return MatrixWrapper(other - self.matrix)
        else:
            raise TypeError(f"Unsupported type for reverse subtraction: {type(other)}")

    # 定义输出格式
    def __repr__(self):
        return f"MatrixWrapper(\n{self.matrix})"


# 测试加减法操作
A = MatrixWrapper([[1, 2], [3, 4]])
B = MatrixWrapper([[5, 6], [7, 8]])

# 两个 MatrixWrapper 实例的加减
print("A + B =\n", A + B)
print("A - B =\n", A - B)

# MatrixWrapper 与 NumPy 矩阵的加减
C = np.array([[10, 20], [30, 40]])
print("A + C =\n", A + C)
print("C - A =\n", C - A)