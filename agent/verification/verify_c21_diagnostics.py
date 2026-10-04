"""Generate reproducible reference fixtures with ArviZ 0.22.0 (verification only).

Run with py312 and --reference-path pointing to an isolated pip --target
installation of arviz==0.22.0, xarray==2025.1.2, xarray-einstats==0.8.0.
The package and ordinary test suite do not depend on these reference libraries.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def makeCases():
    rng = np.random.default_rng(20260921)
    cases = {}
    for chains in (2, 4):
        for draws in (4, 6, 8, 31, 128, 513):
            cases[f'normal_{chains}_{draws}'] = rng.normal(size=(chains, draws))
    for rho in (-.95, -.5, .5, .95):
        for draws in (64, 513, 1024):
            samples = rng.normal(size=(4, draws))
            for draw in range(1, draws):
                samples[:, draw] = rho * samples[:, draw - 1] + np.sqrt(1 - rho ** 2) * samples[:, draw]
            cases[f'ar_{rho}_{draws}'] = samples
    for df in (1, 2, 3):
        cases[f'student_t_{df}'] = rng.standard_t(df, size=(4, 513))
    cases['shifted_chain'] = rng.normal(size=(4, 512)) + np.arange(4)[:, None] * 2
    cases['different_scales'] = rng.normal(size=(4, 512)) * np.array([.1, .5, 2, 10])[:, None]
    cases['drift'] = rng.normal(size=(4, 512)) + np.linspace(-3, 3, 512)
    cases['discrete_ties'] = rng.integers(-20, 21, size=(4, 512)).astype(float)
    cases['single_chain'] = rng.normal(size=(1, 128))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-path', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.reference_path).resolve()))
    import arviz as az
    assert az.__version__ == '0.22.0', az.__version__
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from UQPyL.inference.diagnostics import computeChainDiagnostics

    cases = makeCases()
    rows, compared = [], 0
    maxError = 0.
    for name, samples in cases.items():
        expected = {
            'ess_bulk': float(az.ess(samples, method='bulk')),
            'ess_tail': float(az.ess(samples, method='tail')),
        }
        if samples.shape[0] >= 2:
            expected.update(split_rhat=float(az.rhat(samples, method='split')),
                            rhat=float(az.rhat(samples, method='rank')))
        actual = computeChainDiagnostics(samples[:, :, None])
        for key, value in expected.items():
            assert actual[key]['status'] == ['available'], (name, key, actual[key])
            np.testing.assert_allclose(actual[key]['values'], [value], rtol=2e-10, atol=2e-10)
            maxError = max(maxError, abs(actual[key]['values'][0] - value))
            compared += 1
        rows.append({'name': name, 'expected': expected})
    fixtureDir = root / 'tests' / 'data'
    fixtureDir.mkdir(exist_ok=True)
    np.savez_compressed(fixtureDir / 'inference_diagnostics_arviz_022.npz', **cases)
    report = {'reference': 'ArviZ 0.22.0', 'cases': rows}
    (fixtureDir / 'inference_diagnostics_arviz_022.json').write_text(json.dumps(report, indent=2) + '\n')
    evidence = {'reference': report['reference'], 'cases': len(rows), 'metrics_compared': compared,
                'max_absolute_difference': maxError, 'rtol': 2e-10, 'atol': 2e-10,
                'reference_is_runtime_dependency': False}
    (root / 'agent/verification/0919-c21-reference.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
