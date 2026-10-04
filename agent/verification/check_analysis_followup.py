"""Independent numerical evidence for the three 2026-10-01 followups."""

from decimal import Decimal, localcontext
from pathlib import Path
import json
import warnings

import numpy as np
from scipy.spatial.distance import cdist

from UQPyL.analysis import DeltaTest, Morris, MARS
from UQPyL.doe import LHS, MorrisDesign
from UQPyL.problem import Problem


OUTPUT = Path(__file__).with_name("1001-analysis-followup.json")


def deltaReference(x, y, mask):
    # Exhaustive distances and direct differences; no analysis helper calls.
    distances = cdist(x[:, mask], x[:, mask])
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :2]
    return float(0.5 * np.mean((y[:, None, :] - y[neighbors]) ** 2))


def checkDelta():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: 3 * x[:, :1] + 1)
    x = LHS("classic").sample(problem, 128, seed=17)
    baseY = problem.evaluate(x).objs
    masks = [[True, False], [False, True], [True, True]]
    records = []
    for factor, constantColumn in [
        (1, False),
        (1e-200, False),
        (-1e-200, False),
        (1e200, False),
        (1e-200, True),
        (1e200, True),
    ]:
        y = baseY * factor
        referenceY = baseY
        if constantColumn:
            y = np.column_stack([y, np.full(len(x), 1e300)])
            referenceY = np.column_stack([baseY, np.zeros(len(x))])
        objectives = [deltaReference(x, referenceY, mask) for mask in masks]
        assert np.argmin(objectives) == 0
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always", RuntimeWarning)
            method = DeltaTest(verboseFlag=False)
            labels = method.findCombVio(problem, x, y)
            result = method.findCombEA(problem, x, y, FEs=50, seed=17, verboseFlag=False, saveFlag=False)
        assert labels == [problem.xLabels[0]]
        np.testing.assert_array_equal(result.bestDecs, [[1, 0]])
        scale = result.extra["delta_selection"]
        with localcontext() as context:
            context.prec = 80
            amplitude = Decimal.from_float(scale["scale_mantissa"]) * Decimal(2) ** scale["scale_exponent"]
            actual = Decimal.from_float(float(result.bestObjs[0, 0])) * amplitude**2
            expected = Decimal.from_float(objectives[0]) * Decimal.from_float(float(factor)) ** 2
            relativeError = float(abs(actual / expected - 1))
        assert relativeError < 1e-10
        records.append(
            dict(
                factor=factor,
                constant_column=constantColumn,
                selected_labels=labels,
                ea_mask=result.bestDecs.tolist(),
                objective_metadata=scale,
                physical_objective_decimal=str(actual),
                reference_decimal=str(expected),
                relative_error=relativeError,
                warnings=[str(item.message) for item in emitted],
            )
        )
    return records


def checkMorris(numLevels=4):
    problem = Problem(nInput=2, nObj=1, lb=[10.0, -2.0], ub=[30.0, 1.0], objFunc=lambda x: 3 * x[:, :1] - 2 * x[:, 1:2])
    x, meta = MorrisDesign(numLevels=numLevels).sampleWithMeta(problem, 20, seed=17)
    y = problem.evaluate(x).objs
    records = []
    for factor in (0.001, 1, 1000):
        lb, ub = problem.lb.copy(), problem.ub.copy()
        lb[0, 0], ub[0, 0] = lb[0, 0] * factor + 7, ub[0, 0] * factor + 7
        converted = x.copy()
        converted[:, 0] = converted[:, 0] * factor + 7
        convertedProblem = Problem(nInput=2, nObj=1, lb=lb, ub=ub, objFunc=lambda x: x[:, :1])
        expected = [60, -6]
        result = Morris(verboseFlag=False).analyze(convertedProblem, converted, y, meta)
        np.testing.assert_allclose(result["mu"].values[0], expected, rtol=1e-10, atol=1e-12)
        records.append(
            dict(
                input_factor=factor,
                num_levels=numLevels,
                effect_mode="unit",
                effects=result["mu"].values[0].tolist(),
                analytic_effects=expected,
                normalized=result["S1_norm"].values[0].tolist(),
            )
        )
    return records


def checkMars():
    problem = Problem(nInput=5, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])
    x = LHS("classic").sample(problem, 512, seed=41)
    additive = 3 * x[:, 0] + x[:, 1] ** 2
    weakThird = 3 * x[:, 0] + 2 * x[:, 1] * x[:, 2] * x[:, 3]
    records = []
    for name, degree, values, losses in [
        ("additive", 2, additive, [3, 4 / 45, 0, 0, 0]),
        ("weak_third_order", 2, weakThird, [3, 4 / 27, 4 / 27, 4 / 27, 0]),
        ("weak_third_order", 3, weakThird, [3, 4 / 27, 4 / 27, 4 / 27, 0]),
    ]:
        print(f"Checking MARS {name}, degree={degree}, 3 holdouts", flush=True)
        reference = np.array(losses) / sum(losses)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always", RuntimeWarning)
            result = MARS(verboseFlag=False, maxDegree=degree, nValidationRepeats=3).analyze(
                problem, x, values[:, None]
            )
        weights = result["S1_norm"].values[0]
        primary = result.extra["mars_validation"]["outputs"][0]
        stability = result.extra["mars_stability"]
        np.testing.assert_array_equal(weights, stability["splits"][0]["normalized_weights"][0])
        if name == "weak_third_order" and degree == 3:
            assert primary["validation_r2"] > 0.95
            assert primary["gcv_search_unstable"]
            assert np.max(np.abs(weights - reference)) > 0.1
        records.append(
            dict(
                model=name,
                degree=degree,
                samples=len(x),
                sample_seed=41,
                analytic_normalized_deletion_loss=reference.tolist(),
                primary_weights=weights.tolist(),
                primary_max_error=float(np.max(np.abs(weights - reference))),
                primary_mean_error=float(np.mean(np.abs(weights - reference))),
                validation=primary,
                stability=stability,
                warnings=[str(item.message) for item in emitted],
            )
        )
    return records


def main():
    results = dict(delta=checkDelta(), morris=checkMorris(), mars=checkMars())
    OUTPUT.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
