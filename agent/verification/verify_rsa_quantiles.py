"""Compare repaired RSA regions with Decimal quantiles and ordinary NumPy."""

import argparse
from decimal import Decimal, localcontext
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import cramervonmises_2samp

from UQPyL.analysis import RSA
from UQPyL.problem import Problem


OUTPUT = Path(__file__).with_name("1001-analysis-boundary-quantiles.json")


def decimalQuantiles(values, probabilities):
    ordered = sorted(float(value) for value in values)
    with localcontext() as context:
        context.prec = 1100
        quantiles = []
        for probability in probabilities:
            # Use the same binary sample position, but calculate the linear
            # interpolation independently at high precision.
            position = float((len(ordered) - 1) * probability)
            index = int(position)
            fraction = Decimal.from_float(position) - index
            lower = Decimal.from_float(ordered[index])
            upper = Decimal.from_float(ordered[min(index + 1, len(ordered) - 1)])
            quantiles.append(float(lower + fraction * (upper - lower)))
    return np.array(quantiles)


def checkCase(name, values, nRegion, *, checkNumpy):
    probabilities = np.linspace(0, 1, nRegion + 1)
    expected = decimalQuantiles(values, probabilities)
    actual = RSA._outputQuantiles(values, probabilities)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(values[:, None] <= actual, values[:, None] <= expected)
    if checkNumpy:
        reference = np.quantile(values, probabilities)
        np.testing.assert_array_equal(values[:, None] <= actual, values[:, None] <= reference)

    x = np.linspace(0, 1, len(values))[:, None]
    statistics = []
    regionSampleCounts = []
    for region in range(nRegion):
        lowerMask = values >= expected[region] if region == 0 else values > expected[region]
        selected = lowerMask & (values <= expected[region + 1])
        regionSampleCounts.append(int(np.count_nonzero(selected)))
        if np.count_nonzero(selected) >= 2 and np.count_nonzero(~selected) >= 2:
            statistics.append(cramervonmises_2samp(x[selected, 0], x[~selected, 0]).statistic)
    expectedScore = float(np.mean(statistics)) if statistics else 0.0
    problem = Problem(nInput=1, nObj=1, lb=0, ub=1, objFunc=lambda x: x)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = RSA(nRegion=nRegion, verboseFlag=False).analyze(problem, x, values[:, None])
    expectedStatus = (
        "constant_output" if np.all(values == values[0]) else ("estimated" if statistics else "insufficient_samples")
    )
    assert len(emitted) == int(expectedStatus == "insufficient_samples")
    assert all(item.category is RuntimeWarning and "zero placeholders" in str(item.message) for item in emitted)
    output = result.extra["rsa_regions"]["outputs"][0]
    assert output["status"] == expectedStatus
    assert output["valid_region_count"] == len(statistics)
    assert output["region_sample_counts"] == regionSampleCounts
    score = float(result["S1"].values[0, 0])
    np.testing.assert_allclose(score, expectedScore, rtol=1e-14, atol=1e-15)
    return dict(
        name=name,
        sample_count=len(values),
        n_region=nRegion,
        numpy_region_comparison=checkNumpy,
        decimal_regions_match=True,
        expected_score=expectedScore,
        actual_score=score,
        output_status=output["status"],
        valid_region_count=output["valid_region_count"],
        region_sample_counts=regionSampleCounts,
        warning_count=len(emitted),
        max_quantile_absolute_error=float(np.max(np.abs(actual - expected))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    outputPath = parser.parse_args().output
    rng = np.random.default_rng(17)
    records = []
    for count in (4, 16, 65, 256):
        arrays = dict(
            normal=rng.normal(size=count),
            positive=rng.uniform(0.1, 100, count),
            negative=-rng.uniform(0.1, 100, count),
            offset=1e300 + np.arange(count) * 1e285,
            binary=np.array([0.0, 1.0])[np.arange(count) % 2],
        )
        for name, values in arrays.items():
            for nRegion in (2, 5, 20):
                records.append(checkCase(name, values, nRegion, checkNumpy=True))
    values = rng.uniform(-0.5, 0.5, 65)
    for scale in (1e-200, -1e-200, 1e200, 1e308):
        for nRegion in (2, 5, 20):
            records.append(checkCase(f"scale_{scale}", values * scale, nRegion, checkNumpy=True))
    largest, smallest = np.finfo(float).max, np.nextafter(0.0, 1.0)
    extremeArrays = (
        np.array([-largest, -largest, largest, largest]),
        np.array([-largest, -1e-200, 1e-200, largest]),
        np.array([-largest, -1e-200, -smallest, 0.0, smallest, 1e-200, largest]),
    )
    for number, values in enumerate(extremeArrays):
        for nRegion in (2, 5, 20):
            records.append(checkCase(f"extreme_{number}", values, nRegion, checkNumpy=False))
    outputPath.write_text(json.dumps(dict(case_count=len(records), records=records), indent=2, allow_nan=False) + "\n")
    print(f"{len(records)} RSA cases match Decimal regions and independent CvM scores.")
    print(f"{sum(item['numpy_region_comparison'] for item in records)} ordinary cases preserve NumPy regions.")
    print(f"{sum(item['warning_count'] for item in records)} insufficient-region warnings verified.")
    print(f"Saved {outputPath}")


if __name__ == "__main__":
    main()
