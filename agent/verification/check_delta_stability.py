"""Measure conditional screening stability versus sample size and neighbor count."""

import json
import warnings
from pathlib import Path

import numpy as np
from UQPyL.analysis import DeltaTest
from UQPyL.doe import LHS
from UQPyL.problem import Problem

records = []
for dimensions in (3, 10):
    p = Problem(nInput=dimensions, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])
    for size in (128, 512, 2048):
        for seed in (3, 17, 41):
            x = LHS("classic").sample(p, size, seed=seed)
            for noise in (0, 1.5):
                y = (3 * x[:, 0] + 2 * x[:, 1] + noise * np.random.default_rng(seed + 100).normal(size=size))[:, None]
                for neighbors in (1, 2, 5, 10):
                    with warnings.catch_warnings(record=True) as emitted:
                        warnings.simplefilter("always")
                        result = DeltaTest(nNeighbors=neighbors, verboseFlag=False).analyze(p, x, y)
                    score = result["S1"].values[0]
                    records.append(
                        dict(
                            dimensions=dimensions,
                            size=size,
                            seed=seed,
                            noise=noise,
                            neighbors=neighbors,
                            scores=score.tolist(),
                            top_variable=int(np.argmax(score)),
                            top_two=np.argsort(score)[-2:].tolist(),
                            warnings=[str(w.message) for w in emitted],
                        )
                    )
Path(__file__).with_name("0930-delta-stability.json").write_text(json.dumps(records, indent=2) + "\n")
print(len(records), "records")
for dimensions in (3, 10):
    for noise in (0, 1.5):
        for size in (128, 512, 2048):
            rows = [r for r in records if r["dimensions"] == dimensions and r["noise"] == noise and r["size"] == size]
            print(
                dimensions,
                noise,
                size,
                "top0",
                sum(r["top_variable"] == 0 for r in rows),
                "/",
                len(rows),
                "active_top_two",
                sum(set(r["top_two"]) == {0, 1} for r in rows),
            )
