"""Configuration copying shared by the built-in kernel families."""

from copy import deepcopy

import numpy as np

from .setting import Setting


class KernelTemplate:
    # name -> (scalar only, zero allowed, positive infinity allowed).
    _parameterRules = {}

    def _validateParameter(self, name, value, nInput=None):
        scalar, allowZero, allowInfinity = self._parameterRules[name]
        label = "length_scale (l)" if name == "l" else name
        try:
            raw = np.asarray(value)
            if raw.dtype.kind not in "iuf":
                raise ValueError
            array = raw.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must contain real numeric values.") from exc
        if array.ndim > 1 or array.size == 0 or (scalar and array.size != 1):
            shape = "a scalar" if scalar else "a scalar or nonempty one-dimensional vector"
            raise ValueError(f"{label} must be {shape}.")
        valid = np.isfinite(array) | (np.isposinf(array) if allowInfinity else False)
        valid &= array >= 0 if allowZero else array > 0
        if not np.all(valid):
            domain = "nonnegative" if allowZero else "positive"
            extra = " (np.inf is also allowed)" if allowInfinity else ""
            raise ValueError(f"{label} must be finite and {domain}{extra}.")
        if not scalar and nInput is not None and array.size not in (1, nInput):
            raise ValueError(f"{label} must be scalar or have {nInput} values to match input features.")

    def _setKernelParameter(self, name, value, attr=None):
        # Validate before Setting flattens arrays or coerces numeric types.
        self._validateParameter(name, value)
        self.setting.set(name, value, attr)
        self._validateParameterBounds(name)

    def _validateParameterBounds(self, name, nInput=None):
        if name not in self.setting.parVal or self.setting.isChoicePara(name):
            return  # Categorical bounds encode choices, not physical values.
        lower, upper = self.setting.parLB[name], self.setting.parUB[name]
        for bound in (lower, upper):
            self._validateParameter(name, bound, nInput)
            if not np.all(np.isfinite(bound)):
                raise ValueError(f"{name} optimization bounds must be finite.")
            if self.setting.parLog[name] and np.any(bound <= 0):
                raise ValueError(f"{name} logarithmic bounds must be positive.")
        try:
            ordered = np.all(lower <= upper)
        except ValueError as exc:
            raise ValueError(f"{name} optimization bounds have incompatible dimensions.") from exc
        if not ordered:
            raise ValueError(f"{name} lower bound must not exceed its upper bound.")

    def validateParameters(self, nInput=None, checkBounds=False):
        if nInput is not None and (isinstance(nInput, (bool, np.bool_))
                                   or not isinstance(nInput, (int, np.integer)) or nInput <= 0):
            raise ValueError("nInput must be a positive integer.")
        for name in self._parameterRules:
            if self.setting.hasPara(name):
                if self.setting.isChoicePara(name):
                    encoded = self.setting.parVal[name]
                    if (not np.all(np.isfinite(encoded))
                            or np.any(encoded < self.setting.parLB[name])
                            or np.any(encoded > self.setting.parUB[name])):
                        raise ValueError(f"{name} choice coordinates must be finite and within bounds.")
                self._validateParameter(name, self.setting.get(name), nInput)
                if checkBounds:
                    self._validateParameterBounds(name, nInput)

    @staticmethod
    def _checkFeatureMatrix(values, name):
        if not isinstance(values, np.ndarray) or values.ndim != 2 or values.shape[1] == 0:
            raise ValueError(f"{name} must be a two-dimensional array with at least one feature.")
        return values.shape[1]

    def _expandKernelParam(self, name, size=None):
        if name in self.setting.parCon:
            value = np.asarray(self.setting.parCon[name], dtype=float).ravel()
            if size is not None:
                if value.size == 1:
                    value = np.repeat(value, size)
                elif value.size != size:
                    raise ValueError(f"Parameter '{name}' must have size {size}.")
            self.setting.parCon[name] = value.copy()
        else:
            self.setting.expandParam(name, size=size)

    def clone(self):
        """Copy the current kernel configuration without model parameters.

        Built-in kernels only hold configuration. Custom kernels that add
        training caches or external resources should override this method
        to return an independent, untrained kernel.
        """
        names = set(self.getActiveParameters()) - {"kernel"}
        setting = Setting()
        setting.defaultOwner = "kernel"
        for field in ("parVal", "parCon", "parUB", "parLB", "parType",
                      "parSet", "parLog", "parOwner"):
            source = getattr(self.setting, field)
            setattr(setting, field, {name: value for name, value in source.items()
                                     if name in names})
        setting = deepcopy(setting)
        result = deepcopy(self, {id(self.setting): setting})
        result.__dict__.pop("_templateSetting", None)
        return result


def installKernel(model, kernel, kernelFamily):
    # The caller supplies a template; only the private copy is trained.
    if not isinstance(kernel, kernelFamily):
        raise TypeError("kernel must belong to this model's kernel family.")
    internalKernel = kernel.clone()
    oldKernelNames = []
    if model.kernel is not None:
        oldKernelNames = [
            name for name in model.kernel.setting.getParaList(owner="kernel", tunableOnly=False)
            if name != "kernel"
        ]

    kernelChoiceValue = model.setting.parVal.get("kernel", None)
    kernelChoiceAttr = model.setting.parSet.get("kernel", None)
    kernelChoiceOwner = model.setting.parOwner.get("kernel", None)

    if oldKernelNames:
        model.setting.removeParas(oldKernelNames)

    model.kernel = internalKernel
    model.setting.mergeSetting(model.kernel.setting)
    model.kernel.setting = model.setting

    if kernelChoiceValue is not None and kernelChoiceAttr is not None:
        model.setting.parVal["kernel"] = model.setting._normalize_choice_array(kernel, kernelChoiceAttr)
        model.setting.parSet["kernel"] = kernelChoiceAttr
        model.setting.parType["kernel"] = 2
        model.setting.parOwner["kernel"] = kernelChoiceOwner
        model.setting.parLB["kernel"] = np.asarray([0.0])
        model.setting.parUB["kernel"] = np.asarray([float(len(kernelChoiceAttr[0]))])
        model.setting.parLog["kernel"] = False

    if model.xTrain is not None and hasattr(model.kernel, "initialize"):
        model.kernel.initialize(model.xTrain.shape[1])
    model.resetFitState()
    return model
