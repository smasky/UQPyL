"""Check remaining sample/float boundaries after SA01-SA07 repairs."""

from decimal import Decimal, localcontext
import json
from pathlib import Path
import warnings

import numpy as np

from UQPyL.analysis import DeltaTest, Morris, RSA, Sobol
from UQPyL.doe import LHS, SaltelliDesign
from UQPyL.problem import Problem

from review_analysis_module import inspectRun, sanitize


OUTPUT = Path(__file__).with_name("1001-analysis-postfix-review.json")


def reviewRsa():
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    records = []
    for count, regions in ((3, 2), (20, 20), (20, 2), (40, 20), (80, 20)):
        x = np.linspace(0, 1, count)[:, None]
        y = x.copy()
        boundaries = np.quantile(y, np.linspace(0, 1, regions + 1))
        valid = 0
        counts = []
        for region in range(regions):
            lower = y >= boundaries[region] if region == 0 else y > boundaries[region]
            selected = lower & (y <= boundaries[region + 1])
            size = int(np.count_nonzero(selected))
            counts.append(size)
            valid += size >= 2 and count - size >= 2
        records.append(
            dict(
                case="region_budget",
                samples=count,
                n_region=regions,
                region_counts=counts,
                usable_regions=int(valid),
                output_is_constant=False,
                observed=inspectRun(RSA(nRegion=regions, verboseFlag=False), problem, x, y),
            )
        )
    x = np.linspace(0, 1, 20)[:, None]
    for regions in (0, 1, -1, 2.5, True):
        records.append(
            dict(
                case="region_configuration",
                n_region=regions,
                observed=inspectRun(RSA(nRegion=regions, verboseFlag=False), problem, x, x),
            )
        )
    return records


def reviewSobol():
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1] + 2 * x[:, 1:2])
    x, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 128, seed=17)
    y = problem.evaluate(x).objs
    records = [
        dict(
            case="complete_design",
            actual_base_samples=128,
            meta=meta,
            population_s1=[0.2, 0.8],
            observed=inspectRun(Sobol(verboseFlag=False), problem, x, y, meta),
        )
    ]
    records.append(
        dict(
            case="one_complete_block_removed",
            actual_base_samples=127,
            metadata_base_samples=meta["N"],
            observed=inspectRun(Sobol(verboseFlag=False), problem, x[:-6], y[:-6], meta),
        )
    )
    inconsistent = dict(meta, N=1)
    records.append(
        dict(
            case="inconsistent_base_size",
            actual_base_samples=128,
            metadata_base_samples=1,
            observed=inspectRun(Sobol(verboseFlag=False), problem, x, y, inconsistent),
        )
    )
    inconsistent = dict(meta, secondOrder=False)
    records.append(
        dict(
            case="inconsistent_order",
            actual_second_order=True,
            metadata_second_order=False,
            metadata_block_size=meta["blockSize"],
            observed=inspectRun(Sobol(verboseFlag=False), problem, x, y, inconsistent),
        )
    )
    return records


def reviewMorris():
    x = np.array([0, 2 / 3, 1 / 3, 1.0])[:, None]
    meta = dict(designType="morris", numLevels=4, numTrajectory=2)
    records = []
    for scale in (1.0, 1e307, 1e308):
        problem = Problem(
            nInput=1,
            nObj=1,
            lb=0,
            ub=1,
            objFunc=lambda values, scale=scale: np.where(values < 0.5, 0, np.where(values < 0.9, 1, -1)) * scale,
        )
        y = problem.evaluate(x).objs
        with localcontext() as context:
            context.prec = 40
            sigma = Decimal(2).sqrt() * Decimal("1.5") * Decimal.from_float(scale)
            representable = sigma <= Decimal.from_float(np.finfo(float).max)
        records.append(
            dict(
                output_scale=scale,
                analytic_sigma=str(sigma),
                sigma_representable=representable,
                expected="Finite sigma or explicit range diagnosis.",
                observed=inspectRun(Morris(verboseFlag=False), problem, x, y, meta),
            )
        )
    return records


def inspectSelection(method, problem, x, y):
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        try:
            record = dict(status="returned", selection=method.findCombVio(problem, x, y))
        except Exception as error:
            record = dict(status="error", exception_type=type(error).__name__, exception_message=str(error))
    record["warnings"] = [dict(category=item.category.__name__, message=str(item.message)) for item in emitted]
    return record


def reviewDelta():
    unitProblem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1])
    records = []
    for seed in (0, 3, 17, 41):
        unit = LHS("classic").sample(unitProblem, 128, seed=seed)
        physical = unit.copy()
        physical[:, 0] = -0.9 + 0.8 * unit[:, 0]
        y = 3 * unit[:, :1] + 1
        for scale in (1.0, 1e100, 1e308):
            problem = Problem(
                nInput=2,
                nObj=1,
                lb=[-scale, 0],
                ub=[scale, 1],
                objFunc=lambda values, scale=scale: 3 * (values[:, :1] / scale + 0.9) / 0.8 + 1,
            )
            x = physical.copy()
            x[:, 0] *= scale
            records.append(
                dict(
                    case="input_units",
                    sampling_seed=seed,
                    input_scale=scale,
                    active_input=0,
                    observed=inspectRun(DeltaTest(verboseFlag=False), problem, x, y),
                )
            )
            if scale in (1.0, 1e308):
                records.append(
                    dict(
                        case="subset_selection",
                        sampling_seed=seed,
                        input_scale=scale,
                        expected_label=problem.xLabels[0],
                        observed=inspectSelection(DeltaTest(verboseFlag=False), problem, x, y),
                    )
                )
    return records


def main():
    results = dict(rsa=reviewRsa(), sobol=reviewSobol(), morris=reviewMorris(), delta=reviewDelta())
    OUTPUT.write_text(json.dumps(sanitize(results), indent=2, allow_nan=False) + "\n")
    print(f"Saved {OUTPUT}")
    for method, records in results.items():
        for record in records:
            print(json.dumps(sanitize(dict(method=method, **record)), ensure_ascii=False))


if __name__ == "__main__":
    main()
