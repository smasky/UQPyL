"""Compare interaction degrees against independent population deletion references.

For independent U[-1, 1] inputs, removing variable j has optimal prediction
loss D_j = E[Var(Y | X_without_j)]. These references are derived analytically
from the test functions, rather than from another fitted model. MARS uses
positive GCV differences, so agreement is an empirical screening comparison,
not an assertion that its scores equal Sobol indices or population losses.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np

from UQPyL.analysis import MARS
from UQPyL.doe import LHS
from UQPyL.problem import Problem


OUTPUT = Path(__file__).with_name("0930-mars-degree-accuracy.json")


def makeCases(x):
    return {
        "additive": (3 * x[:, 0] + x[:, 1] ** 2, [3, 4 / 45, 0, 0, 0]),
        "second_order": (x[:, 0] * x[:, 1] + 0.2 * x[:, 2], [1 / 9, 1 / 9, 0.04 / 3, 0, 0]),
        "weak_third_order": (
            3 * x[:, 0] + 2 * x[:, 1] * x[:, 2] * x[:, 3],
            [3, 4 / 27, 4 / 27, 4 / 27, 0],
        ),
        "strong_third_order": (
            x[:, 0] + 3 * x[:, 1] * x[:, 2] * x[:, 3],
            [1 / 3, 1 / 3, 1 / 3, 1 / 3, 0],
        ),
    }


def validateReferences():
    # Three-point Gauss-Legendre quadrature integrates these squared polynomial
    # residuals exactly, independently checking the hand-derived formulas.
    nodes, nodeWeights = np.polynomial.legendre.leggauss(3)
    nodeWeights /= 2
    mesh = np.meshgrid(*([nodes] * 5), indexing="ij")
    x = np.column_stack([axis.ravel() for axis in mesh])
    jointWeights = np.prod(np.meshgrid(*([nodeWeights] * 5), indexing="ij"), axis=0)
    records = []
    for name, (values, analytic) in makeCases(x).items():
        gridValues = values.reshape((3,) * 5)
        losses = []
        for variable in range(5):
            axisShape = [1] * 5
            axisShape[variable] = 3
            conditional = np.sum(gridValues * nodeWeights.reshape(axisShape), axis=variable)
            residual = gridValues - np.expand_dims(conditional, variable)
            losses.append(float(np.sum(jointWeights * residual**2)))
        np.testing.assert_allclose(losses, analytic, rtol=1e-12, atol=1e-14)
        records.append(
            dict(
                model=name,
                analytic_losses=analytic,
                quadrature_losses=losses,
                max_absolute_error=float(np.max(np.abs(np.asarray(losses) - analytic))),
            )
        )
    referencePath = OUTPUT.with_name("0930-mars-degree-reference.json")
    referencePath.write_text(json.dumps(records, indent=2) + "\n")
    return records


def summarize(records):
    summaries = []
    for model in ("additive", "second_order", "weak_third_order", "strong_third_order"):
        for size in (128, 512):
            paired = [row for row in records if row["model"] == model and row["samples"] == size]
            improvements = []
            for seed in (3, 17, 41):
                pair = {row["degree"]: row for row in paired if row["seed"] == seed}
                improvements.append(pair[3]["normalized_mae"] < pair[2]["normalized_mae"] - 1e-12)
            for degree in (2, 3):
                selected = [row for row in paired if row["degree"] == degree]
                summaries.append(
                    dict(
                        model=model,
                        samples=size,
                        degree=degree,
                        runs=len(selected),
                        mean_normalized_mae=float(np.mean([row["normalized_mae"] for row in selected])),
                        max_normalized_mae=max(row["normalized_mae"] for row in selected),
                        all_active_above_inactive=sum(row["all_active_above_inactive"] for row in selected),
                        missing_active_variables=sum(len(row["missing_active_variables"]) for row in selected),
                        mean_inactive_mass=float(np.mean([row["inactive_mass"] for row in selected])),
                        mean_pairwise_order_accuracy=float(
                            np.mean([row["pairwise_order_accuracy"] for row in selected])
                        ),
                        mean_validation_r2=float(np.mean([row["validation_r2"] for row in selected])),
                        mean_seconds=float(np.mean([row["seconds"] for row in selected])),
                        degree_three_improves_mae_pairs=sum(improvements),
                    )
                )
    return summaries


def main():
    validateReferences()
    problem = Problem(nInput=5, nObj=1, lb=-1, ub=1, objFunc=lambda x: x[:, :1])
    records = []
    for size in (128, 512):
        for seed in (3, 17, 41):
            x = LHS("classic").sample(problem, size, seed=seed)
            for name, (values, populationLosses) in makeCases(x).items():
                reference = np.asarray(populationLosses, dtype=float)
                reference /= reference.sum()
                active = reference > 0
                pairs = [(i, j) for i in range(5) for j in range(5) if reference[i] > reference[j] + 1e-12]
                for degree in (2, 3):
                    with warnings.catch_warnings(record=True) as emitted:
                        warnings.simplefilter("always", RuntimeWarning)
                        start = time.perf_counter()
                        result = MARS(verboseFlag=False, maxDegree=degree, maxTerms=40).analyze(
                            problem, x, values[:, None]
                        )
                        elapsed = time.perf_counter() - start
                    weights = result["S1_norm"].values[0]
                    diagnostics = result.extra["mars_validation"]["outputs"][0]
                    records.append(
                        dict(
                            model=name,
                            samples=size,
                            seed=seed,
                            degree=degree,
                            normalized=weights.tolist(),
                            reference_normalized=reference.tolist(),
                            reference_population_losses=populationLosses,
                            raw_scores=result["S1"].values[0].tolist(),
                            normalized_mae=float(np.mean(np.abs(weights - reference))),
                            all_active_above_inactive=bool(weights[active].min() > weights[~active].max()),
                            missing_active_variables=np.flatnonzero(active & (weights <= 1e-12)).tolist(),
                            inactive_mass=float(weights[~active].sum()),
                            pairwise_order_accuracy=float(np.mean([weights[i] > weights[j] for i, j in pairs])),
                            validation_r2=diagnostics["validation_r2"],
                            base_gcv=diagnostics["base_gcv"],
                            removed_gcv=diagnostics["removed_gcv"],
                            seconds=elapsed,
                            warnings=[str(item.message) for item in emitted],
                        )
                    )
                    OUTPUT.write_text(json.dumps(dict(records=records), indent=2) + "\n")
                    print(
                        name, size, seed, degree, "MAE", records[-1]["normalized_mae"], "weights", weights, flush=True
                    )
    summaries = summarize(records)
    OUTPUT.write_text(json.dumps(dict(records=records, summaries=summaries), indent=2) + "\n")
    for row in summaries:
        print("SUMMARY", json.dumps(row), flush=True)
    print("Saved", len(records), "fits to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
