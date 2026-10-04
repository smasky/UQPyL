"""Independent repair evidence, preserving the earlier failing audit artifacts."""

import json
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from UQPyL.analysis import DeltaTest, MARS
from UQPyL.doe import LHS
from UQPyL.problem import Problem


OUTPUT = Path(__file__).with_name("0930-delta-ties-range-repair.json")


def referenceDelta(x, y, k):
    distances = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    errors = np.mean((y[:, None, :] - y[None, :, :]) ** 2, axis=2)
    cutoffs = np.sort(distances, axis=1)[:, k - 1]
    # Squared-distance comparisons need twice the relative distance tolerance.
    tied = np.isclose(distances, cutoffs[:, None], rtol=16 * np.finfo(float).eps * x.shape[1], atol=0)
    closer = (distances < cutoffs[:, None]) & ~tied
    strictlyCloser = np.sum(errors * closer, axis=1)
    tiedMean = np.sum(errors * tied, axis=1) / tied.sum(axis=1)
    return float(0.5 * np.mean((strictlyCloser + (k - closer.sum(axis=1)) * tiedMean) / k))


def previousDelta(x, y, k):
    # The preceding implementation is included only as a timing baseline.
    # Its arbitrary tied-neighbor scores are intentionally not a reference.
    _, indices = KDTree(x).query(x, k=k + 1)
    nearest = np.array([row[row != i][:k] for i, row in enumerate(indices)])
    return float(0.5 * np.mean((y[:, None, :] - y[nearest]) ** 2))


def main():
    records = []
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    grid = np.array(list(product(np.arange(4) / 3, repeat=2)))
    continuous = np.random.default_rng(17).random((64, 2))
    for name, x in [("grid", grid), ("duplicates", np.repeat(grid, 3, axis=0)), ("continuous", continuous)]:
        y = (x[:, 0] + 1.2 * x[:, 1])[:, None]
        for k in (1, 2, 5):
            base = referenceDelta(x, y, k)
            expected = np.array([referenceDelta(np.delete(x, j, axis=1), y, k) - base for j in range(2)])
            for seed in range(10):
                order = np.random.default_rng(seed).permutation(len(x))
                result = DeltaTest(nNeighbors=k, verboseFlag=False).analyze(p, x[order], y[order])
                actual = result["S1"].values[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
                records.append(
                    dict(
                        case="permutation",
                        samples=name,
                        neighbors=k,
                        seed=seed,
                        scores=actual.tolist(),
                        reference=expected.tolist(),
                        max_reference_error=float(np.max(np.abs(actual - expected))),
                    )
                )
    print("Permutation references passed:", len(records), flush=True)

    x = LHS("classic").sample(p, 256, seed=17)
    y = (3 * x[:, 0] + 1)[:, None]
    for methodClass in (DeltaTest, MARS):
        baseline = methodClass(verboseFlag=False).analyze(p, x, y)["S1_norm"].values[0]
        for factor in (1, 1e-6, 1e-160, 1e-200, -1e-200, 1e150, 1e155, 1e200):
            row = dict(case="output_scale", method=methodClass.__name__, factor=factor)
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter("always", RuntimeWarning)
                try:
                    result = methodClass(verboseFlag=False).analyze(p, x, y * factor)
                except ValueError as exc:
                    assert abs(factor) >= 1e155 and "finite squared-output range" in str(exc)
                    row.update(error_type=type(exc).__name__, error=str(exc))
                else:
                    assert abs(factor) < 1e155
                    normalized = result["S1_norm"].values[0]
                    np.testing.assert_allclose(normalized, baseline, atol=1e-10, rtol=1e-10)
                    row.update(scores=result["S1"].values[0].tolist(), normalized=normalized.tolist())
            row["warnings"] = [str(item.message) for item in emitted]
            records.append(row)
    print("Output-scale checks passed: 16", flush=True)

    for dimensions in (3, 10):
        p = Problem(nInput=dimensions, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])
        for size in (128, 512, 2048):
            for seed in (3, 17, 41):
                x = LHS("classic").sample(p, size, seed=seed)
                for noise in (0, 1.5):
                    y = (3 * x[:, 0] + 2 * x[:, 1] + noise * np.random.default_rng(seed + 100).normal(size=size))[
                        :, None
                    ]
                    for k in (1, 2, 5, 10):
                        scores = DeltaTest(nNeighbors=k, verboseFlag=False).analyze(p, x, y)["S1"].values[0]
                        assert np.argmax(scores) == 0 and set(np.argsort(scores)[-2:]) == {0, 1}
                        records.append(
                            dict(
                                case="linear_stability",
                                dimensions=dimensions,
                                samples=size,
                                seed=seed,
                                noise=noise,
                                neighbors=k,
                                scores=scores.tolist(),
                            )
                        )
    print("Linear screening checks passed: 144", flush=True)
    OUTPUT.write_text(json.dumps(dict(records=records), indent=2) + "\n")

    timing = []
    rng = np.random.default_rng(17)
    benchmarkCases = [("continuous", rng.random((size, 5))) for size in (128, 2048)]
    benchmarkCases += [("identical", np.zeros((size, 5))) for size in (128, 2048)]
    for name, x in benchmarkCases:
        y = rng.random((len(x), 1))
        oldTimes, newTimes = [], []
        for _ in range(3):
            start = time.perf_counter()
            previousDelta(x, y, 2)
            oldTimes.append(time.perf_counter() - start)
            start = time.perf_counter()
            DeltaTest(verboseFlag=False)._cal_delta(x, y, 2)
            newTimes.append(time.perf_counter() - start)
        oldMedian, newMedian = float(np.median(oldTimes)), float(np.median(newTimes))
        timing.append(
            dict(
                samples=name,
                size=len(x),
                previous_seconds=oldMedian,
                repaired_seconds=newMedian,
                ratio=newMedian / oldMedian,
            )
        )
    OUTPUT.write_text(json.dumps(dict(records=records, timing=timing), indent=2) + "\n")
    print(
        "Maximum permutation reference error:",
        max(row["max_reference_error"] for row in records if row["case"] == "permutation"),
    )
    print("Timing:", json.dumps(timing), flush=True)
    print("Saved", len(records), "verification records to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
