import os


class UQPyLConfig:
    def __init__(self):
        self._workDir = None
        self._resultDirName = "Result"

    @property
    def workDir(self):
        return self._workDir

    @workDir.setter
    def workDir(self, value):
        self._workDir = None if value is None else os.fspath(value)

    @property
    def resultDirName(self):
        return self._resultDirName

    @resultDirName.setter
    def resultDirName(self, value):
        if value is None:
            raise ValueError("resultDirName cannot be None.")
        value = os.fspath(value)
        if value == "":
            raise ValueError("resultDirName cannot be empty.")
        self._resultDirName = value

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(type(self), key):
                raise KeyError(f"Unknown UQPyL config key: {key}")
            setattr(self, key, value)
        return self

    def reset(self):
        self._workDir = None
        self._resultDirName = "Result"
        return self

    def resolveWorkDir(self, *candidates):
        for candidate in candidates:
            if candidate is not None:
                return os.fspath(candidate)
        if self.workDir is not None:
            return self.workDir
        return os.getcwd()


config = UQPyLConfig()
