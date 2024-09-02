def instance_decorator(func):
    def wrapper(self, *args, **kwargs):
        print("Accessing variable before calling the method:", self.variable)
        # 在这里可以调用ClassB的任何其他方法
        result = func(self, *args, **kwargs)
        # 在方法调用后访问变量或处理结果
        print("Accessing variable after calling the method:", self.variable)
        return result
    return wrapper
class ClassB:
    def __init__(self, variable):
        self.variable = variable  # ClassB的一个实例变量

    @instance_decorator
    def some_method(self, x):
        print("Method some_method is being executed with argument:", x)
        # 这里可以进行一些操作，比如修改variable
        self.variable += x
        return self.variable
obj = ClassB(10)
result = obj.some_method(5)
print("The result of some_method:", result)