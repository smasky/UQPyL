"""Attribute native iteration warnings to individual fits without changing SVR."""

import json
import os
import warnings
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
from scipy.stats import qmc

from UQPyL.surrogate import AutoTuner
from UQPyL.surrogate.svr import SVR


records = []
for case, seed in [("local_peak", 11), ("local_peak", 23), ("local_peak", 47), ("discontinuous", 11)]:
    x = qmc.LatinHypercube(2, seed=seed).random(192)
    query = np.random.default_rng(731).uniform(size=(4096, 2))

    def function(values):
        if case == "local_peak":
            return np.exp(-np.sum(((values - 0.37) / 0.06) ** 2, axis=1))[:, None]
        return (values[:, 0] + values[:, 1] > 1).astype(float)[:, None]

    predictions = []
    for limit in [100000, 1000000]:
        model = SVR(maxIter=limit)
        originalFit = model.fitModel
        fits = []

        def trackedFit(trainX, trainY):
            row = dict(
                index=len(fits),
                n_train=len(trainX),
                parameters={
                    name: float(np.asarray(model.setting.get(name)).item()) for name in ["C", "epsilon", "gamma"]
                },
            )
            savedFd = os.dup(2)
            with TemporaryFile() as captured, warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    os.dup2(captured.fileno(), 2)
                    result = originalFit(trainX, trainY)
                finally:
                    os.dup2(savedFd, 2)
                    os.close(savedFd)
                    captured.seek(0)
                    row["native_stderr"] = captured.read().decode().strip()
                    row["python_warnings"] = [str(item.message) for item in caught]
                    row["solver"] = model.fitState.get("solver")
                    fits.append(row)
            return result

        model.fitModel = trackedFit
        tuner = AutoTuner(model)
        _, validation = tuner.gridTune(
            x,
            function(x),
            paraGrid={"C": np.log([1, 100]), "epsilon": np.log([0.001, 0.1]), "gamma": np.log([1, 10, 100])},
            ratio=25,
            seed=seed,
            tuneMode="joint",
        )
        prediction = model.predict(query)
        predictions.append(prediction)
        truth = function(query)
        record = dict(
            case=case,
            seed=seed,
            max_iter=limit,
            fits=fits,
            selected_candidate=tuner.getReport()["best_candidate_index"],
            validation_r2=float(validation),
            test_r2=float(1 - np.sum((prediction - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)),
            fit_state_keys=list(model.fitState),
            final_solver=tuner.getReport()["final_refit"].get("solver"),
        )
        if len(predictions) == 2:
            record["max_prediction_change"] = float(np.max(np.abs(predictions[1] - predictions[0])))
        records.append(record)
        print(
            case,
            seed,
            limit,
            "warned fits",
            [r["index"] for r in fits if r["native_stderr"]],
            "selected",
            record["selected_candidate"],
            "test R2",
            record["test_r2"],
            flush=True,
        )

Path(__file__).with_name("1002-svr-iteration-limit-after.json").write_text(json.dumps(records, indent=2) + "\n")
