"""Audit evidence only: expose limitations without changing production semantics."""

import json
from pathlib import Path
import numpy as np
from UQPyL.analysis import MARS, Morris, DeltaTest
from UQPyL.surrogate.mars import MARS as MarsModel
from UQPyL.doe import LHS, MorrisDesign
from UQPyL.problem import Problem

quiet = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def problem(bounds=1):
    return Problem(nInput=3, nObj=1, lb=-np.ones(3) * bounds, ub=np.ones(3) * bounds, objFunc=lambda x: x[:, :1])


def main():
    records = []
    # Same physical response, same sample rows; only x0's numerical unit changes.
    p = problem()
    x, meta = MorrisDesign().sampleWithMeta(p, 32, seed=17)
    y = (x[:, 0] + 2 * x[:, 1])[:, None]
    for factor in (1, 0.001, 1000):
        scale = np.array([factor, 1, 1])
        r = Morris(**quiet).analyze(problem(scale), x * scale, y, meta=meta)
        np.testing.assert_allclose(r["mu"].values[0], [1 / factor, 2, 0], atol=1e-10)
        records.append(
            dict(
                case="morris_units", factor=factor, mu=r["mu"].values[0].tolist(), norm=r["S1_norm"].values[0].tolist()
            )
        )
    for seed in (3, 17, 41, 73, 101):
        x = LHS("classic").sample(p, 512, seed=seed)
        y = (x[:, 0] + 2 * x[:, 1])[:, None]
        for factor in (1, 0.001, 1000):
            scale = np.array([factor, 1, 1])
            r = DeltaTest(**quiet).analyze(problem(scale), x * scale, y)
            records.append(
                dict(
                    case="delta_units",
                    seed=seed,
                    factor=factor,
                    scores=r["S1"].values[0].tolist(),
                    norm=r["S1_norm"].values[0].tolist(),
                )
            )
        for size in (512, 2048):
            x = LHS("classic").sample(p, size, seed=seed)
            testX = np.random.default_rng(seed + 1000).uniform(-1, 1, (4096, 3))
            for name, objective in (
                ("additive", lambda a: a[:, 0] ** 2 + 2 * a[:, 1]),
                ("product", lambda a: a[:, 0] * a[:, 1]),
                ("interaction_plus_main", lambda a: a[:, 0] * a[:, 1] + 0.1 * a[:, 2]),
            ):
                y = objective(x)[:, None]
                target = objective(testX)[:, None]
                r = MARS(**quiet).analyze(p, x, y)
                row = dict(
                    case="mars_models",
                    seed=seed,
                    size=size,
                    model=name,
                    scores=r["S1"].values[0].tolist(),
                    norm=r["S1_norm"].values[0].tolist(),
                )
                for degree in (1, 2):
                    model = MarsModel(max_degree=degree, max_terms=400 if degree == 1 else 40)
                    model.fit(x, y)
                    row[f"r2_degree_{degree}"] = float(
                        1 - np.mean((model.predict(testX) - target) ** 2) / np.var(target)
                    )
                records.append(row)
                print(row, flush=True)
    # Balanced independent factorial sample: no finite-sample marginal signal.
    x = np.array(np.meshgrid(*([np.linspace(-1, 1, 9)] * 3), indexing="ij")).reshape(3, -1).T
    y = (x[:, 0] * x[:, 1])[:, None]
    r = MARS(**quiet).analyze(p, x, y)
    records.append(
        dict(case="mars_balanced_product", scores=r["S1"].values[0].tolist(), norm=r["S1_norm"].values[0].tolist())
    )
    output = Path(__file__).with_name("0930-analysis-recheck.json")
    output.write_text(json.dumps(records, indent=2) + "\n")
    print("Saved", len(records), "records to", output)
    for row in records:
        if row["case"] != "mars_models":
            print(row)


if __name__ == "__main__":
    main()
