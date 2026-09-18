"""Compare deterministic MARS fits before and after a native rebuild."""

import argparse
import json
from pathlib import Path

import numpy as np

from UQPyL.surrogate.mars import MARS


def collectResults():
    results = {}
    for seed in (7, 19):
        rng = np.random.default_rng(seed)
        trainX = rng.uniform(-2., 2., (80, 2))
        testX = rng.uniform(-2., 2., (31, 2))
        trainY = np.column_stack([
            np.maximum(trainX[:, 0] - .2, 0.) + .5 * trainX[:, 1],
            np.sin(trainX[:, 0]) + trainX[:, 0] * trainX[:, 1],
        ])
        for smooth in (False, True):
            for prune in (False, True):
                model = MARS(max_terms=15, max_degree=2, smooth=smooth,
                             enable_pruning=prune)
                model.fit(trainX, trainY)
                key = f'{seed}_{smooth}_{prune}'
                results[key + '_predictions'] = model.predict(testX)
                results[key + '_coefficients'] = model.coef_.copy()
                results[key + '_basis'] = model.transform(testX)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['save', 'compare'])
    parser.add_argument('baseline', type=Path)
    args = parser.parse_args()
    results = collectResults()
    if args.mode == 'save':
        np.savez(args.baseline, **results)
        print(json.dumps({'cases': 8, 'arrays': len(results), 'baseline': str(args.baseline)}))
    else:
        with np.load(args.baseline) as baseline:
            assert set(results) == set(baseline.files)
            maxDifference = 0.
            for key, value in results.items():
                np.testing.assert_allclose(value, baseline[key], rtol=1e-12, atol=1e-12)
                maxDifference = max(maxDifference, float(np.max(np.abs(value - baseline[key]))))
        print(json.dumps({'cases': 8, 'arrays': len(results), 'max_absolute_difference': maxDifference}))
