"""Affine input-unit and output-unit checks with an independent distance oracle."""

import json
from pathlib import Path

import numpy as np
from UQPyL.analysis import DeltaTest
from UQPyL.doe import LHS
from UQPyL.problem import Problem


def pairwiseDelta(x, y):
    distances = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :2]
    return 0.5 * np.mean((y[:, None, :] - y[nearest]) ** 2)


def main():
    records = []
    for seed in (3, 17, 41, 73, 101):
        p = Problem(nInput=3, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])
        x = LHS("classic").sample(p, 512, seed=seed)
        y = (x[:, 0] + 2 * x[:, 1])[:, None]
        unit = (x + 1) / 2
        base = pairwiseDelta(unit, y)
        expected = np.array([pairwiseDelta(np.delete(unit, j, axis=1), y) - base for j in range(3)])
        normalized = expected / np.sum(np.abs(expected))
        for factor in (1, 0.001, 1000):
            scale, shift = np.array([factor, 1, 1]), np.array([10, -3, 20])
            transformed = Problem(nInput=3, nObj=1, lb=-scale + shift, ub=scale + shift, objFunc=lambda x: x[:, :1])
            for outputFactor in (1, 1e-6):
                result = DeltaTest(verboseFlag=False).analyze(transformed, x * scale + shift, y * outputFactor)
                raw = result["S1"].values[0]
                actual = result["S1_norm"].values[0]
                np.testing.assert_allclose(raw / outputFactor**2, expected, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(actual, normalized, rtol=1e-12, atol=1e-14)
                assert np.argmax(raw) == np.argmax(actual) == 1
                records.append(
                    dict(
                        seed=seed,
                        input_factor=factor,
                        output_factor=outputFactor,
                        raw_scores=raw.tolist(),
                        normalized_scores=actual.tolist(),
                        max_reference_error=float(np.max(abs(raw / outputFactor**2 - expected))),
                    )
                )
    path = Path(__file__).with_name("0930-delta-scaling.json")
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"{len(records)} comparisons passed; max reference error:", max(r["max_reference_error"] for r in records))
    print("seed=17:", [r for r in records if r["seed"] == 17 and r["output_factor"] == 1])


if __name__ == "__main__":
    main()
