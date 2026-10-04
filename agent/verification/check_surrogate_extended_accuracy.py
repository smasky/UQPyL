"""Stress prediction accuracy on independent points; no production changes."""

import json
from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryFile
from time import perf_counter
import warnings

import numpy as np
from scipy.stats import qmc

from UQPyL.surrogate import AutoTuner, MinMaxScaler, StandardScaler
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as GpRbf
from UQPyL.surrogate.gp.kernel import Matern
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.mars import MARS
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.regression import LinearRegression, PolynomialRegression
from UQPyL.surrogate.svr import SVR


def makeModel(name):
    factories = {
        "LR": LinearRegression,
        "PR2": PolynomialRegression,
        "GPR": GPR,
        "KRG": KRG,
        "RBF": RBF,
        "MARS": MARS,
        "SVR": SVR,
        "SVR_tuned": SVR,
        "GPR_ard": lambda: GPR(kernel=GpRbf(heterogeneous=True)),
        "GPR_noise": lambda: GPR(C_attr={"lb": 1e-8, "ub": 1, "type": "float", "log": True}),
        "GPR_matern": lambda: GPR(kernel=Matern()),
        "GPR_y_scaled": lambda: GPR(scalers=(None, StandardScaler())),
        "MARS3": lambda: MARS(max_degree=3),
        "GPR_scaled": lambda: GPR(scalers=(MinMaxScaler(), StandardScaler())),
        "RBF_scaled": lambda: RBF(scalers=(MinMaxScaler(), StandardScaler())),
        "KRG_scaled": lambda: KRG(scalers=(MinMaxScaler(), StandardScaler())),
        "SVR_scaled_tuned": lambda: SVR(scalers=(MinMaxScaler(), StandardScaler())),
    }
    return factories[name]()


cases = {
    "anisotropic": (2, lambda x: np.sin(12 * np.pi * x[:, 0]) + 0.1 * x[:, 1], ["GPR_ard"]),
    "sparse8d": (8, lambda x: np.sin(2 * np.pi * x[:, 0]) + 0.5 * x[:, 1] ** 2, ["GPR_ard"]),
    "local_peak": (2, lambda x: np.exp(-np.sum(((x - 0.37) / 0.06) ** 2, axis=1)), ["GPR_matern", "GPR_y_scaled"]),
    "discontinuous": (2, lambda x: (x[:, 0] + x[:, 1] > 1).astype(float), ["GPR_matern", "GPR_noise"]),
    "triple_interaction": (3, lambda x: np.prod(2 * x - 1, axis=1), ["MARS3", "GPR_ard"]),
    "noisy": (1, lambda x: np.sin(2 * np.pi * x[:, 0]), ["GPR_noise"]),
    "physical_units": (
        2,
        lambda x: np.sin(2 * np.pi * x[:, 0]) + 0.5 * np.cos(2 * np.pi * x[:, 1]),
        ["GPR_scaled", "KRG_scaled", "RBF_scaled", "SVR_scaled_tuned"],
    ),
    "extrapolation": (1, lambda x: np.sin(2 * np.pi * x[:, 0]), []),
}


def metrics(truth, prediction):
    residual = prediction - truth
    return dict(
        r2=float(1 - np.sum(residual**2) / np.sum((truth - truth.mean()) ** 2)),
        nrmse=float(np.sqrt(np.mean(residual**2)) / np.std(truth)),
        max_abs_error=float(np.max(np.abs(residual))),
    )


def serialize(value):
    if isinstance(value, np.ndarray):
        return serialize(value.tolist())
    if isinstance(value, np.generic):
        return serialize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize(item) for item in value]
    return value


@contextmanager
def captureNativeStderr(row):
    # LIBSVM emits iteration-limit messages directly to file descriptor 2.
    # Preserve them per fit, including candidates inside AutoTuner.
    savedFd = os.dup(2)
    with TemporaryFile() as captured:
        try:
            os.dup2(captured.fileno(), 2)
            yield
        finally:
            os.dup2(savedFd, 2)
            os.close(savedFd)
            captured.seek(0)
            row["native_stderr"] = captured.read().decode(errors="replace").strip()


records = []
output = Path(__file__).with_name("1002-surrogate-extended-accuracy.json")
for case, (dimension, function, extras) in cases.items():
    unitTest = np.random.default_rng(731).uniform(size=(4096, dimension))
    # Both sides of the training interval; separate from in-domain predictions.
    if case == "extrapolation":
        unitTest = np.concatenate([np.linspace(-0.25, 0, 2048, endpoint=False), np.linspace(1.0001, 1.25, 2048)])[
            :, None
        ]
    trueTest = function(unitTest)[:, None]
    factor = np.array([1e-3, 1e3]) if case == "physical_units" else np.ones(dimension)
    testX = unitTest * factor
    for nTrain in [64, 192]:
        for seed in [11, 23, 47]:
            unitTrain = qmc.LatinHypercube(dimension, seed=seed).random(nTrain)
            trainX = unitTrain * factor
            trainY = function(unitTrain)[:, None]
            if case == "noisy":
                trainY = trainY + np.random.default_rng(seed + 1000).normal(0, 0.1, trainY.shape)
            for name in ["LR", "PR2", "GPR", "KRG", "RBF", "MARS", "SVR", "SVR_tuned", *extras]:
                row = dict(case=case, n_train=nTrain, seed=seed, model=name)
                model = makeModel(name)
                model.rng = np.random.default_rng(seed)
                start = perf_counter()
                with warnings.catch_warnings(record=True) as caught, captureNativeStderr(row):
                    warnings.simplefilter("always")
                    try:
                        if name.endswith("tuned"):
                            tuner = AutoTuner(model)
                            _, validationScore = tuner.gridTune(
                                trainX,
                                trainY,
                                paraGrid={
                                    "C": np.log([1, 100]),
                                    "epsilon": np.log([0.001, 0.1]),
                                    "gamma": np.log([1, 10, 100]),
                                },
                                ratio=25,
                                seed=seed,
                                tuneMode="joint",
                            )
                            row["validation_r2"] = float(validationScore)
                        else:
                            model.fit(trainX, trainY)
                        pred = model.predict(testX)
                        row.update(metrics(trueTest, pred))
                        row["train_rmse"] = float(np.sqrt(np.mean((model.predict(trainX) - trainY) ** 2)))
                        if case == "local_peak":
                            center = np.full((1, dimension), 0.37)
                            row["peak_prediction"] = float(model.predict(center).item())
                        if case == "extrapolation":
                            inside = np.linspace(0, 1, 1001)[:, None]
                            row["inside"] = metrics(function(inside)[:, None], model.predict(inside))
                        if name.startswith("GPR") or name.startswith("KRG"):
                            _, std = model.predict(testX, returnStd=True)
                            row["latent_coverage_95"] = float(np.mean(np.abs(pred - trueTest) <= 1.96 * std))
                            row["mean_std"] = float(np.mean(std))
                        if name.startswith("GPR"):
                            row["length_scale"] = model.kernel.setting.get("l")
                            row["noise_variance"] = model.setting.get("C")
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                row["warnings"] = [str(item.message) for item in caught]
                row["seconds"] = perf_counter() - start
                records.append(serialize(row))
        output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(case, nTrain, "completed", flush=True)

for case in cases:
    names = list(dict.fromkeys(r["model"] for r in records if r["case"] == case))
    for name in names:
        rows = [r for r in records if r["case"] == case and r["model"] == name and r["n_train"] == 192]
        print(
            case,
            name,
            "R2",
            [round(r.get("r2", -999), 5) for r in rows],
            "coverage",
            [round(r.get("latent_coverage_95", -1), 3) for r in rows],
            flush=True,
        )
