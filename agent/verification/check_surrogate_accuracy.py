"""Independent prediction audit; run with py312, PYTHONPATH=. and one BLAS thread."""

import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import qmc

from UQPyL.surrogate import AutoTuner
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as GpRbf
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.mars import MARS
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.regression import LinearRegression, PolynomialRegression
from UQPyL.surrogate.svr import SVR


def makeModel(name):
    factories = {
        "LR": LinearRegression,
        "PR2": PolynomialRegression,
        "RBF": RBF,
        "KRG": KRG,
        "GPR": GPR,
        "GPR_old": lambda: GPR(kernel=GpRbf(length_attr={"lb": 1, "ub": 1e5, "type": "float", "log": True})),
        "GPR_wide": lambda: GPR(kernel=GpRbf(length_attr={"lb": 0.01, "ub": 1e5, "type": "float", "log": True})),
        "MARS": MARS,
        "MARS1": lambda: MARS(max_degree=1),
        "MARS2": lambda: MARS(max_degree=2),
        "SVR": SVR,
        "SVR_tuned": SVR,
        "SVR_configured": lambda: SVR(C=100, epsilon=0.001, gamma=10),
    }
    return factories[name]()


functions = {
    "linear": (2, lambda x: 1 + 2 * x[:, 0] - x[:, 1]),
    "quadratic": (2, lambda x: x[:, 0] ** 2 + 2 * x[:, 0] * x[:, 1] - x[:, 1] ** 2),
    "smooth": (2, lambda x: np.sin(2 * np.pi * x[:, 0]) + 0.5 * np.cos(2 * np.pi * x[:, 1])),
    "oscillatory": (1, lambda x: np.sin(8 * np.pi * x[:, 0])),
    "interaction": (2, lambda x: (2 * x[:, 0] - 1) * (2 * x[:, 1] - 1)),
}
names = [
    "LR",
    "PR2",
    "RBF",
    "KRG",
    "GPR",
    "GPR_old",
    "GPR_wide",
    "MARS",
    "MARS1",
    "MARS2",
    "SVR",
    "SVR_configured",
    "SVR_tuned",
]
records = []
for case, (dimension, function) in functions.items():
    testX = np.random.default_rng(712).uniform(size=(4096, dimension))
    testY = function(testX)[:, None]
    for nTrain in [32, 128]:
        for seed in [11, 23, 47]:
            trainX = qmc.LatinHypercube(dimension, seed=seed).random(nTrain)
            trainY = function(trainX)[:, None]
            for name in names:
                model = makeModel(name)
                model.rng = np.random.default_rng(seed)
                row = dict(case=case, n_train=nTrain, seed=seed, model=name)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        if name == "SVR_tuned":
                            tuner = AutoTuner(model)
                            _, score = tuner.gridTune(
                                trainX,
                                trainY,
                                paraGrid={
                                    "C": np.log([1, 100]),
                                    "epsilon": np.log([0.001, 0.01]),
                                    "gamma": np.log([1, 10, 100]),
                                },
                                ratio=25,
                                seed=seed,
                                tuneMode="joint",
                            )
                            row["validation_r2"] = float(score)
                            row["selected_parameters"] = {
                                key: np.asarray(model.setting.get(key)).tolist() for key in ["C", "epsilon", "gamma"]
                            }
                        else:
                            model.fit(trainX, trainY)
                        pred = model.predict(testX)
                        residual = pred - testY
                        row.update(
                            r2=float(1 - np.sum(residual**2) / np.sum((testY - testY.mean()) ** 2)),
                            nrmse=float(np.sqrt(np.mean(residual**2)) / np.std(testY)),
                            max_abs_error=float(np.max(np.abs(residual))),
                            train_rmse=float(np.sqrt(np.mean((model.predict(trainX) - trainY) ** 2))),
                        )
                        if name.startswith("GPR") or name == "KRG":
                            _, std = model.predict(testX, returnStd=True)
                            row["coverage_95"] = float(np.mean(np.abs(residual) <= 1.96 * std))
                            row["mean_std"] = float(np.mean(std))
                        if name.startswith("GPR"):
                            row["length_scale"] = np.asarray(model.kernel.setting.get("l")).tolist()
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                row["warnings"] = [str(item.message) for item in caught]
                records.append(row)
        print(case, nTrain, "completed", flush=True)

output = Path(__file__).with_name("1002-surrogate-accuracy-after-fixes.json")
output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
for case in functions:
    for name in names:
        rows = [r for r in records if r["case"] == case and r["model"] == name and r["n_train"] == 128]
        print(
            case,
            name,
            "R2",
            [round(r.get("r2", -999), 6) for r in rows],
            "coverage",
            [round(r.get("coverage_95", -1), 3) for r in rows],
        )
