import abc
import numpy as np
import warnings
from ._numeric import centeredColumns


class Scaler(metaclass=abc.ABCMeta):
    """Column-wise scaling. One-dimensional inputs represent a single row."""

    def __init__(self):
        self.fitted = False

    def _checkArray(self, values, fitting=False):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] == 0 or (fitting and not len(values)):
            raise ValueError("Scaler expects a nonempty feature matrix (n_samples, n_features).")
        if not np.all(np.isfinite(values)):
            raise ValueError("Scaler values must be finite.")
        if not fitting:
            if not self.fitted:
                raise RuntimeError("Scaler must be fitted before transforming values.")
            if values.shape[1] != self.nFeatures:
                raise ValueError("Scaler feature count does not match fitted data.")
        return values

    @abc.abstractmethod
    def fit(self, trainX):
        pass

    @abc.abstractmethod
    def transform(self, trainX):
        pass

    def fit_transform(self, trainX):
        self.fit(trainX)
        return self.transform(trainX)

    @abc.abstractmethod
    def inverse_transform(self, trainX):
        pass

    def inverse_transform_std(self, values):
        raise NotImplementedError("This scaler does not implement standard-deviation inversion.")

    def inverse_transform_var(self, values):
        raise NotImplementedError("This scaler does not implement variance inversion.")


class _AffineScaler(Scaler):
    """Keep source and target origins separate to preserve constant columns."""

    def _storeAffine(self, values, sourceOrigin, inverseScale, targetOrigin=0.0):
        if not np.all(np.isfinite(sourceOrigin)) or not np.all(np.isfinite(inverseScale)) or np.any(inverseScale <= 0):
            raise ValueError("Scaler fitted offsets and scales must be finite, with positive scales.")
        self.nFeatures = values.shape[1]
        self.sourceOrigin = np.asarray(sourceOrigin)
        self.targetOrigin = targetOrigin
        self.inverseScale = np.asarray(inverseScale)
        self.fitted = True
        return self

    def transform(self, trainX):
        values = self._checkArray(trainX)
        with np.errstate(over="ignore"):
            difference = values - self.sourceOrigin
            result = difference / self.inverseScale
        # Opposite finite extremes can overflow subtraction even when the
        # standardized difference is representable.
        overflow = ~np.isfinite(difference)
        if np.any(overflow):
            with np.errstate(over="ignore", invalid="ignore"):
                alternate = values / self.inverseScale - self.sourceOrigin / self.inverseScale
            result = np.where(overflow, alternate, result)
        return result + self.targetOrigin

    def inverse_transform(self, trainX):
        values = self._checkArray(trainX)
        shifted = values - self.targetOrigin
        with np.errstate(over="ignore", invalid="ignore"):
            result = shifted * self.inverseScale + self.sourceOrigin
        if np.any(~np.isfinite(result)):
            fractions, powers = np.frexp(shifted)
            scaleFractions, scalePowers = np.frexp(self.inverseScale)
            originFractions, originPowers = np.frexp(self.sourceOrigin)
            powers = powers + scalePowers
            commonPower = np.maximum(powers, originPowers)
            with np.errstate(over="ignore", under="ignore"):
                alternate = np.ldexp(
                    np.ldexp(fractions * scaleFractions, powers - commonPower)
                    + np.ldexp(originFractions, originPowers - commonPower),
                    commonPower,
                )
            result = np.where(np.isfinite(result), result, alternate)
        return result

    def inverse_transform_std(self, values):
        values = self._checkArray(values)
        if np.any(values < 0):
            raise ValueError("Standard deviations must be nonnegative.")
        return self._restoreUncertainty(values, squared=False)

    def inverse_transform_var(self, values):
        values = self._checkArray(values)
        if np.any(values < 0):
            raise ValueError("Variances must be nonnegative.")
        return self._restoreUncertainty(values, squared=True)

    def _restoreUncertainty(self, values, *, squared):
        fractions, powers = np.frexp(values)
        scaleFraction, scalePower = np.frexp(self.inverseScale)
        degree = 2 if squared else 1
        with np.errstate(over="ignore", under="ignore"):
            result = np.ldexp(fractions * scaleFraction**degree, powers + degree * scalePower)
        if np.any(~np.isfinite(result)) or np.any((values > 0) & (result == 0)):
            quantity = "variance" if squared else "standard deviation"
            warnings.warn(
                f"Restored {quantity} exceeds floating-point range; underflow is returned as zero "
                "and overflow as infinity. Request returnStd=True when the variance is unrepresentable.",
                RuntimeWarning,
                stacklevel=3,
            )
        return result


def _finiteScalar(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar.")
    return float(value)


class MinMaxScaler(_AffineScaler):
    def __init__(self, min_: float = 0, max_: float = 1):
        super().__init__()
        self.min_scale = _finiteScalar(min_, "min_")
        self.max_scale = _finiteScalar(max_, "max_")
        if self.max_scale <= self.min_scale or not np.isfinite(self.max_scale - self.min_scale):
            raise ValueError("MinMaxScaler requires a finite positive target range.")

    def fit(self, trainX):
        self.fitted = False
        values = self._checkArray(trainX, fitting=True)
        self.min_ = np.min(values, axis=0)
        self.max_ = np.max(values, axis=0)
        span = self.max_ - self.min_
        # A constant training column uses unit source span, retaining invertibility.
        span = np.where(span == 0, 1.0, span)
        inverseScale = span / (self.max_scale - self.min_scale)
        return self._storeAffine(values, self.min_, inverseScale, self.min_scale)


class StandardScaler(_AffineScaler):
    def __init__(self, muX: float = 0, sitaX: float = 1):
        super().__init__()
        self.muX = _finiteScalar(muX, "muX")
        self.sitaX = _finiteScalar(sitaX, "sitaX")
        if self.sitaX <= 0:
            raise ValueError("sitaX must be positive.")

    def fit(self, trainX):
        self.fitted = False
        values = self._checkArray(trainX, fitting=True)
        centered, powers, mean = centeredColumns(values)
        self.mu = np.ldexp(mean, powers)
        # Compute sample deviations in bounded units before restoring their
        # scale; never form the original-unit variance as an intermediate.
        scaledStd = (
            np.sqrt(np.sum(centered**2, axis=0) / (len(values) - 1)) if len(values) > 1 else np.zeros(values.shape[1])
        )
        with np.errstate(over="ignore", under="ignore"):
            self.sita = np.ldexp(scaledStd, powers)
        if np.any((self.sita == 0) & np.any(values != values[0], axis=0)) or not np.all(np.isfinite(self.sita)):
            raise ValueError("StandardScaler standard deviation is outside floating-point range.")
        scale = np.where(self.sita == 0, 1.0, self.sita)
        inverseScale = scale / self.sitaX
        return self._storeAffine(values, self.mu, inverseScale, self.muX)
