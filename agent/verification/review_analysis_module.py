"""Audit sensitivity edge cases without modifying production code or pytest."""

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import scipy

from UQPyL.analysis import FAST, Morris, RBDFAST, RSA, Sobol
from UQPyL.doe import FASTDesign, LHS, MorrisDesign, SaltelliDesign
from UQPyL.problem import Problem


OUTPUT = Path(__file__).with_name("1001-analysis-module-review.json")


def inspectRun(method, problem, x, y=None, meta=None):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        try:
            result = method.analyze(problem, x, y, meta=meta)
            record = dict(
                status="returned",
                metrics={metric.name: metric.values.tolist() for metric in result.metrics},
                input_dtype=str(result.X.dtype),
                output_dtype=str(result.Y.dtype),
            )
        except Exception as error:
            record = dict(status="error", exception_type=type(error).__name__, exception_message=str(error))
    record["warnings"] = [dict(category=item.category.__name__, message=str(item.message)) for item in emitted]
    return record


def reviewMorris():
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: (x > 0.5).astype(np.uint8))
    x, meta = MorrisDesign().sampleWithMeta(problem, 8, seed=17)
    # Every trajectory crosses the threshold, with a signed unit step of 2/3.
    expected = dict(mu=1.5, mu_star=1.5, sigma=0.0)
    records = []
    for dtype in (np.float64, np.uint8, np.bool_):
        y = (x > 0.5).astype(dtype)
        records.append(
            dict(
                case="provided_output",
                dtype=str(y.dtype),
                expected=expected,
                observed=inspectRun(Morris(verboseFlag=False), problem, x, y, meta),
            )
        )
    records.append(
        dict(
            case="evaluated_uint8_output",
            expected=expected,
            observed=inspectRun(Morris(verboseFlag=False), problem, x, meta=meta),
        )
    )
    integerProblem = Problem(nInput=1, nObj=1, lb=0, ub=2, varType=[1], objFunc=lambda x: x.astype(float))
    integerX, integerMeta = MorrisDesign().sampleWithMeta(integerProblem, 8, seed=17)
    for dtype in (np.float64, np.uint8):
        values = integerX.astype(dtype)
        records.append(
            dict(
                case="integer_input",
                dtype=str(values.dtype),
                expected_mu=2.0,
                observed=inspectRun(
                    Morris(verboseFlag=False), integerProblem, values, values.astype(float), integerMeta
                ),
            )
        )
    singleX, singleMeta = MorrisDesign().sampleWithMeta(problem, 1, seed=3)
    single = dict(
        expected="Sample sigma requires at least two effects; report insufficiency explicitly.",
        observed=inspectRun(Morris(verboseFlag=False), problem, singleX, singleX, singleMeta),
    )
    return dict(dtype_cases=records, one_trajectory=single)


def reviewRbdFast():
    fixedProblem = Problem(nInput=2, nObj=1, lb=[0, 0], ub=[0, 1], objFunc=lambda x: x[:, 1:2])
    x = np.column_stack([np.zeros(256), np.linspace(0, 1, 256)])
    fixed = []
    for seed in (None, 0, 17, 41):
        order = np.arange(len(x)) if seed is None else np.random.default_rng(seed).permutation(len(x))
        fixed.append(
            dict(
                permutation_seed=seed,
                expected_first_index=0.0,
                observed=inspectRun(RBDFAST(verboseFlag=False), fixedProblem, x[order], x[order, 1:2]),
            )
        )
    discreteProblem = Problem(nInput=2, nObj=1, lb=0, ub=1, varType=[1, 0], objFunc=lambda x: x[:, 1:2])
    x = np.column_stack([np.repeat([0, 1], 128), np.tile(np.linspace(0, 1, 128), 2)])
    # A Cartesian product: both discrete groups have exactly the same Y distribution.
    discrete = []
    for seed in (None, 0, 17, 41):
        order = np.arange(len(x)) if seed is None else np.random.default_rng(seed).permutation(len(x))
        discrete.append(
            dict(
                permutation_seed=seed,
                population_indices=[0, 1],
                observed=inspectRun(RBDFAST(verboseFlag=False), discreteProblem, x[order], x[order, 1:2]),
            )
        )
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    budget = []
    for count, harmonics in ((8, 4), (8, 5), (16, 12)):
        x = np.random.default_rng(17).random((count, 2))
        budget.append(
            dict(
                samples=count,
                harmonics=harmonics,
                bias_factor=2 * harmonics / count,
                expected="Reject a singular/nonpositive bias-correction denominator.",
                observed=inspectRun(RBDFAST(M=harmonics, verboseFlag=False), problem, x, x[:, 1:2]),
            )
        )
    x = LHS("classic").sample(problem, 1024, seed=17)
    control = inspectRun(RBDFAST(verboseFlag=False), problem, x, x[:, 1:2])
    return dict(fixed_input=fixed, repeated_input=discrete, harmonic_budget=budget, continuous_control=control)


def reviewRsa():
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    x = np.linspace(0, 1, 4)[:, None]
    y = np.array([-1.0, -1.0, 1.0, 1.0])[:, None]
    # Ranks 1,2 vs 3,4: U=16, nm(n+m)=16, (4nm-1)/(6(n+m))=15/24.
    reference = 1 - 15 / 24
    finite = [
        dict(
            factor=factor,
            expected_statistic=reference,
            observed=inspectRun(RSA(nRegion=2, verboseFlag=False), problem, x, y * factor),
        )
        for factor in (1, 1e-200, 1e308)
    ]
    invalid = []
    for value in (np.nan, np.inf, -np.inf):
        changed = y.copy()
        changed[0, 0] = value
        invalid.append(
            dict(
                case="nonfinite_output",
                value=value,
                expected="Explicit rejection/diagnosis",
                observed=inspectRun(RSA(nRegion=2, verboseFlag=False), problem, x, changed),
            )
        )
    changedX = x.copy()
    changedX[0, 0] = np.nan
    invalid.append(
        dict(
            case="nonfinite_input",
            expected="Explicit rejection/diagnosis",
            observed=inspectRun(RSA(nRegion=2, verboseFlag=False), problem, changedX, y),
        )
    )
    return dict(output_scaling=finite, invalid_data=invalid)


def reviewFast():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, 1:2])
    x, meta = FASTDesign(M=4).sampleWithMeta(problem, 129, seed=3)
    y = problem.evaluate(x).objs
    records = [
        dict(
            case="complete_design",
            rows=len(x),
            expected_indices=[0, 1],
            observed=inspectRun(FAST(verboseFlag=False), problem, x, y, meta),
        )
    ]
    for dropped in (1, 2):
        records.append(
            dict(
                case="missing_rows",
                dropped=dropped,
                rows=len(x) - dropped,
                metadata_expected_rows=meta["N"] * problem.nInput,
                expected="Reject incomplete/inconsistent FAST blocks.",
                observed=inspectRun(FAST(verboseFlag=False), problem, x[:-dropped], y[:-dropped], meta),
            )
        )
    records.append(
        dict(
            case="extra_row",
            rows=len(x) + 1,
            metadata_expected_rows=len(x),
            expected="Reject inconsistent FAST blocks; do not silently ignore a row.",
            observed=inspectRun(
                FAST(verboseFlag=False), problem, np.vstack([x, x[-1:]]), np.vstack([y, [[1e200]]]), meta
            ),
        )
    )
    return records


def reviewSobol():
    threshold = 0.85
    problem = Problem(
        nInput=2,
        nObj=1,
        lb=0,
        ub=1,
        objFunc=lambda x: ((x[:, 0] > threshold) & (x[:, 1] > threshold)).astype(float)[:, None],
    )
    records = []
    for seed in range(100):
        x, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, 16, seed=seed)
        y = problem.evaluate(x).objs
        base = np.concatenate([y[::4], y[3::4]])
        if np.var(base) == 0 and np.var(y) > 0:
            records.append(
                dict(
                    case="base_variance_zero",
                    samples_per_base=16,
                    seed=seed,
                    base_variance=float(np.var(base)),
                    all_sample_variance=float(np.var(y)),
                    hybrid_events=int(np.sum(y)),
                    expected="Explicit insufficient-base-variance diagnosis.",
                    observed=inspectRun(Sobol(verboseFlag=False), problem, x, y, meta),
                )
            )
            break
    assert records, "Expected to locate the small-sample rare-event case."
    probability = 1 - threshold
    # Independent Bernoulli conjunction: S1=q/(1+q), ST=1/(1+q).
    x, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, 4096, seed=17)
    records.append(
        dict(
            case="adequate_sample_control",
            samples_per_base=4096,
            analytic_first_order=probability / (1 + probability),
            analytic_total_order=1 / (1 + probability),
            observed=inspectRun(Sobol(verboseFlag=False), problem, x, meta=meta),
        )
    )
    return records


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main(outputPath=OUTPUT):
    results = dict(
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
        morris=reviewMorris(),
        rbd_fast=reviewRbdFast(),
        rsa=reviewRsa(),
        fast=reviewFast(),
        sobol=reviewSobol(),
    )
    outputPath.write_text(json.dumps(sanitize(results), indent=2, allow_nan=False) + "\n")
    print(f"Saved {outputPath}")
    for name in ("morris", "rbd_fast", "rsa", "fast", "sobol"):
        print(f"Recorded {name} edge cases and independent controls.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="Path for the audit JSON; existing data are replaced."
    )
    main(parser.parse_args().output)
