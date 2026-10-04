"""Zero-noise gain comparison; run with OPENBLAS_NUM_THREADS=1 in py312."""

import json
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from UQPyL.calibration.methods._ensemble import anomalyGain


def main():
    rows = []
    rng = np.random.default_rng(29)
    for nEns, nObs in [(32, 24), (16, 128), (24, 512)]:
        x, y = rng.normal(size=(nEns, 3)), rng.normal(size=(nEns, nObs))
        x -= x.mean(axis=0)
        y -= y.mean(axis=0)
        for lam in (0., .1):
            dense = lambda: anomalyGain(x, y, r=np.zeros((nObs, nObs)), lam=lam)[0]
            reduced = lambda: anomalyGain(x, y, lam=lam)[0]
            reference, actual = dense(), reduced()
            np.testing.assert_allclose(actual, reference, rtol=2e-9, atol=2e-11)
            record = {'ensemble_members': nEns, 'observations': nObs, 'lam': lam,
                      'max_abs_difference': float(np.max(np.abs(reference - actual))),
                      'default_solver': anomalyGain(x, y, lam=lam)[1]['solver']}
            for name, function in [('dense', dense), ('default', reduced)]:
                times = []
                for _ in range(5):
                    start = perf_counter()
                    function()
                    times.append(perf_counter() - start)
                tracemalloc.start()
                function()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                record[name + '_median_seconds'] = float(np.median(times))
                record[name + '_traced_peak_bytes'] = peak
            rows.append(record)
    report = {'scope': 'Gain calculation only; tracemalloc peak excludes input arrays and may exclude native workspace. Not process RSS.',
              'records': rows}
    Path('agent/verification/0919-c15-zero-noise-benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
