"""Stress tests distinguish estimator limits from permutation/numerical defects."""

import json
import warnings
from pathlib import Path

import numpy as np
from UQPyL.analysis import DeltaTest, MARS
from UQPyL.doe import LHS
from UQPyL.problem import Problem

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)
OUTPUT = Path(__file__).with_name("0930-analysis-remaining.json")


def tieAveragedDelta(x, y, k):
    # Expected k-neighbor squared error under uniform selection at the cutoff.
    distances = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    cutoffs = np.sort(distances, axis=1)[:, k - 1]
    squared = (y[:, None, :] - y[None, :, :]) ** 2
    errors = []
    for i, cutoff in enumerate(cutoffs):
        closer = distances[i] < cutoff
        tied = distances[i] == cutoff
        slots = k - np.count_nonzero(closer)
        errors.append((squared[i, closer].sum() + slots * squared[i, tied].mean()) / k)
    return 0.5 * np.mean(errors)


def analyze(cls, p, x, y, **kwargs):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = cls(**QUIET, **kwargs).analyze(p, x, y)
    return result, [str(w.message) for w in emitted]


def save(records):
    OUTPUT.write_text(json.dumps(records, indent=2) + "\n")


def main():
    records = []
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    grid = np.array(np.meshgrid(np.arange(4) / 3, np.arange(4) / 3, indexing="ij")).reshape(2, -1).T
    random = np.random.default_rng(17).random((64, 2))
    for name, x in [("grid", grid), ("duplicates", np.repeat(grid, 3, axis=0)), ("continuous", random)]:
        y = (x[:, 0] + 1.2 * x[:, 1])[:, None]
        for k in (1, 2, 5):
            reference = tieAveragedDelta(x, y, k)
            referenceScores = [tieAveragedDelta(np.delete(x, j, axis=1), y, k) - reference for j in range(2)]
            for seed in range(10):
                order = np.random.default_rng(seed).permutation(len(x))
                result, messages = analyze(DeltaTest, p, x[order], y[order], nNeighbors=k)
                records.append(
                    dict(
                        case="delta_permutation",
                        model=name,
                        k=k,
                        seed=seed,
                        scores=result["S1"].values[0].tolist(),
                        reference_scores=referenceScores,
                        delta=DeltaTest(**QUIET)._cal_delta(x[order], y[order], k),
                        tie_average_delta=float(reference),
                        warnings=messages,
                    )
                )
    save(records)
    print("Delta permutations complete:", len(records), flush=True)
    # Small-scale physical outputs still contain distinct, finite values. Check
    # whether normalized importance survives even when squared units underflow.
    x = LHS("classic").sample(p, 256, seed=17)
    y = (3 * x[:, 0] + 1)[:, None]
    for cls in (DeltaTest, MARS):
        for factor in (1, 1e-6, 1e-160, 1e-200, 1e100, 1e155, 1e200):
            row = dict(case="output_scale", method=cls.__name__, factor=factor)
            try:
                result, messages = analyze(cls, p, x, y * factor)
                row.update(
                    scores=result["S1"].values[0].tolist(),
                    normalized=result["S1_norm"].values[0].tolist(),
                    warnings=messages,
                )
            except (ValueError, OverflowError, FloatingPointError) as exc:
                row.update(error_type=type(exc).__name__, error=str(exc))
            records.append(row)
    save(records)
    print("Scale checks complete", flush=True)
    for seed in (3, 17, 41):
        for size in (128, 512):
            p = Problem(nInput=5, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])
            x = LHS("classic").sample(p, size, seed=seed)
            models = {
                "additive": (3 * x[:, 0] + x[:, 1] ** 2, [0, 1]),
                "interaction": (x[:, 0] * x[:, 1] + 0.2 * x[:, 2], [0, 1, 2]),
                "third_order": (3 * x[:, 0] + 2 * x[:, 1] * x[:, 2] * x[:, 3], [0, 1, 2, 3]),
                "noisy": (
                    3 * x[:, 0] + 2 * x[:, 1] + 0.2 * np.random.default_rng(seed + 200).normal(size=size),
                    [0, 1],
                ),
            }
            for name, (values, active) in models.items():
                for split in (0, 1):
                    order = np.arange(size) if split == 0 else np.random.default_rng(91).permutation(size)
                    result, messages = analyze(MARS, p, x[order], values[order, None])
                    score = result["S1_norm"].values[0]
                    records.append(
                        dict(
                            case="mars_stability",
                            seed=seed,
                            size=size,
                            model=name,
                            split=split,
                            active=active,
                            scores=score.tolist(),
                            validation_r2=result.extra["mars_validation"]["outputs"][0]["validation_r2"],
                            warnings=messages,
                        )
                    )
                print("MARS", seed, size, name, records[-1]["validation_r2"], records[-1]["scores"], flush=True)
                save(records)
    # Changing degree is a user-configurable method choice, evaluated on a
    # previously problematic third-order case with the same samples and split.
    p = Problem(nInput=5, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])
    x = LHS("classic").sample(p, 512, seed=17)
    y = (3 * x[:, 0] + 2 * x[:, 1] * x[:, 2] * x[:, 3])[:, None]
    for degree in (2, 3):
        result, messages = analyze(MARS, p, x, y, maxDegree=degree)
        records.append(
            dict(
                case="mars_degree",
                degree=degree,
                scores=result["S1_norm"].values[0].tolist(),
                diagnostics=result.extra,
                warnings=messages,
            )
        )
    # Redundant correlated inputs have no unique drop-variable information:
    # zero raw importance is a property of the conditional estimator here.
    x = np.column_stack((np.linspace(0, 1, 32), np.linspace(0, 1, 32)))
    p = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    result, messages = analyze(DeltaTest, p, x, x[:, :1])
    records.append(dict(case="delta_redundancy", scores=result["S1"].values[0].tolist(), warnings=messages))
    save(records)
    print("Saved", len(records), "records to", OUTPUT)


if __name__ == "__main__":
    main()
