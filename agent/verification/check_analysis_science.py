"""Deterministic multi-seed comparisons against analytic sensitivity targets."""

import json
from pathlib import Path
import numpy as np
from UQPyL.analysis import Sobol, FAST, RBDFAST, Morris, RSA, DeltaTest, MARS
from UQPyL.doe import SaltelliDesign, FASTDesign, LHS, MorrisDesign
from UQPyL.problem import Problem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)


def main():
    records = []
    variance = 0.5 + 49 / 8 + 0.1 * np.pi**4 / 5 + 0.01 * np.pi**8 / 18
    ishS1 = np.array([0.5 * (1 + 0.1 * np.pi**4 / 5) ** 2, 49 / 8, 0]) / variance
    interaction = 0.01 * np.pi**8 * (1 / 18 - 1 / 50) / variance
    models = {
        "linear": (lambda x: (x[:, 0] + 2 * x[:, 1])[:, None], 1.0, [0.2, 0.8, 0], [0.2, 0.8, 0]),
        "product": (lambda x: (x[:, 0] * x[:, 1])[:, None], 1.0, [0, 0, 0], [1, 1, 0]),
        "ishigami": (
            lambda x: (np.sin(x[:, 0]) + 7 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0]))[:, None],
            np.pi,
            ishS1,
            ishS1 + [interaction, 0, interaction],
        ),
    }
    for name, (objective, bound, s1, st) in models.items():
        problem = Problem(nInput=3, nObj=1, lb=-bound, ub=bound, objFunc=objective)
        for methodClass in (Sobol, FAST, RBDFAST):
            for seed in (3, 17, 41, 73, 101):
                if methodClass is Sobol:
                    X, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 8192, seed=seed)
                elif methodClass is FAST:
                    X, meta = FASTDesign(M=4).sampleWithMeta(problem, 4097, seed=seed)
                else:
                    X, meta = LHS("classic").sample(problem, 8192, seed=seed), None
                result = methodClass(**QUIET).analyze(problem, X, meta=meta)
                row = dict(
                    model=name,
                    method=methodClass.__name__,
                    seed=seed,
                    samples=len(X),
                    s1=result["S1"].values[0].tolist(),
                    expected_s1=np.asarray(s1).tolist(),
                )
                row["s1_max_error"] = float(np.max(np.abs(result["S1"].values[0] - s1)))
                if methodClass is not RBDFAST:
                    row.update(
                        st=result["ST"].values[0].tolist(),
                        expected_st=np.asarray(st).tolist(),
                        st_max_error=float(np.max(np.abs(result["ST"].values[0] - st))),
                    )
                if methodClass is Sobol:
                    expected = (
                        [1.0, 0.0, 0.0]
                        if name == "product"
                        else ([0.0, interaction, 0.0] if name == "ishigami" else [0.0, 0.0, 0.0])
                    )
                    row.update(
                        s2=result["S2"].values[0].tolist(),
                        expected_s2=expected,
                        s2_max_error=float(np.max(np.abs(result["S2"].values[0] - expected))),
                    )
                records.append(row)
    # Rank-based methods have their own meaning; do not compare them to Sobol S1.
    for seed in (3, 17, 41, 73, 101):
        p = Problem(nInput=3, nObj=1, lb=0, ub=1, objFunc=lambda x: (x[:, 0] + 2 * x[:, 1])[:, None])
        x, meta = MorrisDesign().sampleWithMeta(p, 32, seed=seed)
        result = Morris(**QUIET).analyze(p, x, meta=meta)
        records.append(
            dict(
                method="Morris",
                seed=seed,
                mu=result["mu"].values[0].tolist(),
                max_error=float(np.max(np.abs(result["mu"].values[0] - [1, 2, 0]))),
                sigma=result["sigma"].values[0].tolist(),
            )
        )
        x = LHS("classic").sample(p, 512, seed=seed)
        # A single active input is a conservative ranking check, not a claim of variance decomposition.
        y = (3 * x[:, 0] + 1)[:, None]
        for cls in (RSA, DeltaTest, MARS):
            result = cls(**QUIET).analyze(p, x, Y=y)
            score = result["S1"].values[0]
            records.append(
                dict(method=cls.__name__, seed=seed, scores=score.tolist(), top_variable=int(np.argmax(score)))
            )
    for row in records:
        if row["method"] in ("Sobol", "FAST", "RBDFAST"):
            tolerance = 0.03 if row["method"] == "Sobol" else 0.06
            assert all(
                row[key] <= tolerance for key in ("s1_max_error", "st_max_error", "s2_max_error") if key in row
            ), row
        elif row["method"] == "Morris":
            assert row["max_error"] < 1e-12, row
        else:
            assert row["top_variable"] == 0, row
    output = Path(__file__).with_name("0929-analysis-science.json")
    output.write_text(json.dumps(records, indent=2) + "\n")
    for method in ("Sobol", "FAST", "RBDFAST"):
        for model in models:
            rows = [r for r in records if r["method"] == method and r.get("model") == model]
            print(
                method,
                model,
                {
                    key: max(r[key] for r in rows)
                    for key in ("s1_max_error", "st_max_error", "s2_max_error")
                    if key in rows[0]
                },
            )
    for method in ("Morris", "RSA", "DeltaTest", "MARS"):
        print(method, [r for r in records if r["method"] == method])


if __name__ == "__main__":
    main()
