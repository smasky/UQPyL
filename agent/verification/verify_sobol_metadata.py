"""Recheck the recorded Sobol metadata failures and unchanged valid metrics."""

import json
from pathlib import Path
import warnings

import numpy as np

from UQPyL.analysis import Sobol
from UQPyL.doe import SaltelliDesign
from UQPyL.problem import Problem


def main():
    folder = Path(__file__).parent
    previous = json.loads((folder / "1001-analysis-postfix-review.json").read_text())["sobol"]
    problem = Problem(nInput=2, nObj=1, lb=0, ub=1, objFunc=lambda x: x[:, :1] + 2 * x[:, 1:2])
    x, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 128, seed=17)
    y = problem.evaluate(x).objs
    fullMetrics = previous[0]["observed"]["metrics"]
    remainingMetrics = previous[1]["observed"]["metrics"]
    cases = [
        ("complete_design", x, y, meta, "validated", fullMetrics),
        ("whole_block_removed", x[:-6], y[:-6], meta, "recovered", remainingMetrics),
        ("wrong_N", x, y, dict(meta, N=1), "recovered", fullMetrics),
        ("wrong_second_order", x, y, dict(meta, secondOrder=False), "recovered", fullMetrics),
        ("updated_N", x[:-6], y[:-6], dict(meta, N=127), "validated", remainingMetrics),
        ("partial_block_removed", x[:-1], y[:-1], meta, "not_estimated", None),
        ("partial_block_added", np.vstack([x, x[:1]]), np.vstack([y, y[:1]]), meta, "not_estimated", None),
    ]
    records = []
    for name, samples, outputs, samplingMeta, expectedStatus, expectedMetrics in cases:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            result = Sobol(verboseFlag=False).analyze(problem, samples, outputs, samplingMeta)
        diagnostic = result.extra["sobol_design"]
        assert diagnostic["status"] == expectedStatus
        assert len(emitted) == int(expectedStatus != "validated")
        assert all(item.category is RuntimeWarning for item in emitted)
        metrics = {metric.name: metric.values.tolist() for metric in result.metrics}
        if expectedMetrics is not None:
            assert metrics == expectedMetrics
        else:
            assert diagnostic["metrics_available"] is False
            assert all(np.all(metric.values == 0) for metric in result.metrics)
        assert result.meta == samplingMeta
        np.testing.assert_array_equal(result.X, samples)
        np.testing.assert_array_equal(result.Y, outputs)
        records.append(dict(case=name, diagnostic=diagnostic, metrics=metrics, warning_count=len(emitted)))

    outputPath = folder / "1001-sobol-warning-reference.json"
    outputPath.write_text(json.dumps(dict(case_count=len(records), records=records), indent=2, allow_nan=False) + "\n")
    print("Seven Sobol controls pass: two validated, three recovered with warnings, two not estimated with warnings.")
    print("All computed metrics exactly match the saved valid complete-block baselines.")
    print("Wrong secondOrder now recovers the actual design instead of returning the old incorrect scores.")
    print(f"Saved {outputPath}")


if __name__ == "__main__":
    main()
