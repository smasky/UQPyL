class ParentClass:
    def decorator(self, func):
        def wrapper(*args, **kwargs):
            print("Accessing parent instance variable:", self.instance_var)
            print("Something is happening before the method is called.")
            result = func(*args, **kwargs)
            print("Something is happening after the method is called.")
            print("Accessing child instance variable (if available):", getattr(self, 'child_var', 'Not Available'))
            return result
        return wrapper

    def __init__(self, value):
        self.instance_var = value

class ChildClass(ParentClass):
    def __init__(self, parent_value, child_value):
        super().__init__(parent_value)
        self.child_var = child_value

    @ParentClass.decorator
    def say_hello(self, name):
        print(f"Hello, {name}!")

# 使用
obj = ChildClass("parent_value", "child_value")
obj.say_hello("Alice")