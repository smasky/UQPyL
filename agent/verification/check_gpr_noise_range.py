"""Paired old/default noise ranges, independent clean targets, fixed data seeds."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from UQPyL.surrogate.gp import GPR

records = []
testX = np.linspace(0, 1, 4096)[:, None]
truth = np.sin(2 * np.pi * testX)
for nTrain in [64, 192]:
    for noiseStd in [0, 0.03, 0.1, 0.3]:
        for seed in [11, 23, 47]:
            x = qmc.LatinHypercube(1, seed=seed).random(nTrain)
            y = np.sin(2 * np.pi * x) + np.random.default_rng(seed + 1000).normal(0, noiseStd, x.shape)
            for name in ["old", "default"]:
                model = GPR(C_attr={"lb": 1e-12, "ub": 1e-6, "type": "float", "log": True}) if name == "old" else GPR()
                model.rng = np.random.default_rng(seed)
                model.fit(x, y)
                prediction, std = model.predict(testX, returnStd=True)
                error = prediction - truth
                records.append(
                    dict(
                        n_train=nTrain,
                        noise_std=noiseStd,
                        seed=seed,
                        configuration=name,
                        r2=float(1 - np.sum(error**2) / np.sum((truth - truth.mean()) ** 2)),
                        rmse=float(np.sqrt(np.mean(error**2))),
                        fitted_noise_variance=float(np.asarray(model.setting.get("C")).item()),
                        latent_coverage_95=float(np.mean(np.abs(error) <= 1.96 * std)),
                    )
                )
output = Path(__file__).with_name("1002-gpr-noise-range.json")
output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
for noiseStd in [0, 0.03, 0.1, 0.3]:
    for name in ["old", "default"]:
        rows = [
            row
            for row in records
            if row["n_train"] == 192 and row["noise_std"] == noiseStd and row["configuration"] == name
        ]
        print(noiseStd, name, "R2", [row["r2"] for row in rows], "C", [row["fitted_noise_variance"] for row in rows])
