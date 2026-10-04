"""Validate repaired screening on analytic interaction cases and independent rows."""

import json
import warnings
from pathlib import Path
import numpy as np
from UQPyL.analysis import MARS
from UQPyL.doe import LHS
from UQPyL.problem import Problem

rows = []
for seed in (3, 17, 41, 73, 101):
    for size in (512, 2048):
        p = Problem(nInput=3, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])
        x = LHS("classic").sample(p, size, seed=seed)
        for name, y in (
            ("additive", x[:, 0] ** 2 + 2 * x[:, 1]),
            ("product", x[:, 0] * x[:, 1]),
            ("mixed", x[:, 0] * x[:, 1] + 0.1 * x[:, 2]),
        ):
            row = dict(seed=seed, size=size, model=name)
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always", RuntimeWarning)
                r = MARS(verboseFlag=False).analyze(p, x, y[:, None])
                row.update(scores=r["S1_norm"].values[0].tolist(), diagnostics=r.extra)
                if name != "additive":
                    assert min(row["scores"][:2]) > row["scores"][2], row
                row["status"] = "warned" if emitted else "accepted"
                row["warnings"] = [str(w.message) for w in emitted]
            rows.append(row)
            print(row, flush=True)
Path("agent/verification/0930-mars-repair.json").write_text(json.dumps(rows, indent=2) + "\n")
