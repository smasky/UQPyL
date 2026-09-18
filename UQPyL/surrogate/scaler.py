import abc
import numpy as np


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
    """Represent inverse scaling as y = offset + inverseScale * z."""
    def _storeAffine(self, values, offset, inverseScale):
        if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(inverseScale)) or np.any(inverseScale <= 0):
            raise ValueError("Scaler fitted offsets and scales must be finite, with positive scales.")
        self.nFeatures = values.shape[1]
        self.offset = np.asarray(offset)
        self.inverseScale = np.asarray(inverseScale)
        self.fitted = True
        return self

    def transform(self, trainX):
        values = self._checkArray(trainX)
        return (values - self.offset) / self.inverseScale

    def inverse_transform(self, trainX):
        values = self._checkArray(trainX)
        return values * self.inverseScale + self.offset

    def inverse_transform_std(self, values):
        values = self._checkArray(values)
        if np.any(values < 0):
            raise ValueError("Standard deviations must be nonnegative.")
        return values * np.abs(self.inverseScale)

    def inverse_transform_var(self, values):
        values = self._checkArray(values)
        if np.any(values < 0):
            raise ValueError("Variances must be nonnegative.")
        return values * self.inverseScale**2


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
        if self.max_scale <= self.min_scale or not np.isfinite(self.max_scale-self.min_scale):
            raise ValueError("MinMaxScaler requires a finite positive target range.")

    def fit(self, trainX):
        self.fitted = False
        values = self._checkArray(trainX, fitting=True)
        self.min_ = np.min(values, axis=0)
        self.max_ = np.max(values, axis=0)
        span = self.max_ - self.min_
        # A constant training column uses unit source span, retaining invertibility.
        span = np.where(span == 0, 1.0, span)
        inverseScale = span / (self.max_scale-self.min_scale)
        return self._storeAffine(values, self.min_-self.min_scale*inverseScale, inverseScale)


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
        self.mu = np.mean(values, axis=0)
        # Preserve sample-standard-deviation semantics for ordinary datasets.
        self.sita = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])
        scale = np.where(self.sita == 0, 1.0, self.sita)
        inverseScale = scale / self.sitaX
        return self._storeAffine(values, self.mu-self.muX*inverseScale, inverseScale)
