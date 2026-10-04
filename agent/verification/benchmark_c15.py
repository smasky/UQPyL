"""Compare the previous eig+solve path with decomposition reuse (py312)."""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from UQPyL.calibration.methods._ensemble import ensembleGain


def previousGain(cxy, cyy, r, lam):
    matrix = cyy + r + lam * np.eye(len(r))
    matrix = (matrix + matrix.T) * .5
    values, vectors = np.linalg.eigh(matrix)
    cutoff = len(matrix) * np.finfo(float).eps * np.max(np.abs(values), initial=0.)
    active = values > cutoff
    if np.count_nonzero(active) == len(matrix):
        return np.linalg.solve(matrix, cxy.T).T
    basis = vectors[:, active]
    return ((cxy @ basis) / values[active]) @ basis.T


def main():
    rng = np.random.default_rng(20260919)
    records = []
    for nObs in (80, 240, 480):
        x = rng.normal(size=(32, 5))
        y = rng.normal(size=(32, nObs))
        x -= x.mean(axis=0)
        y -= y.mean(axis=0)
        cxy, cyy = x.T @ y / 31, y.T @ y / 31
        r = np.eye(nObs) * .5
        expected = previousGain(cxy, cyy, r, .2)
        actual = ensembleGain(cxy, cyy, r, .2)[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
        timing = {}
        for name, function in [('previous', previousGain), ('current', ensembleGain)]:
            times = []
            for _ in range(5):
                start = perf_counter()
                function(cxy, cyy, r, .2)
                times.append(perf_counter() - start)
            timing[name + '_median_seconds'] = float(np.median(times))
        records.append({
            'observations': nObs, 'ensemble_members': 32, 'parameters': 5,
            'max_abs_difference': float(np.max(np.abs(actual - expected))),
            'one_dense_matrix_bytes': int(cyy.nbytes),
            'decompositions_per_full_rank_update_before': 2,
            'decompositions_per_full_rank_update_after': 1,
            **timing,
        })
    output = {
        'scope': 'Synthetic gain calculation, one BLAS thread; not end-to-end or peak memory.',
        'dense_memory_optimization': False,
        'records': records,
    }
    Path('agent/verification/0919-c15-benchmark.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
