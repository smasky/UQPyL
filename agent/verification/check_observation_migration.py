"""Compare fixed-seed calibration arrays across the observation interface migration."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from UQPyL.problem import ModelProblem
from UQPyL.calibration import GLUE, SUFI2, ES, IES

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', action='store_true')
args = parser.parse_args()
arrays = {}
for cls in (GLUE, SUFI2, ES, IES):
    for masked in (False, True):
        obs = np.arange(1., 5.) * .2
        mask = np.array([False, True, False, False]) if masked else None
        def simulate(X):
            sims = X * np.arange(1., 5.)[None, :]
            if masked:
                sims[:, 1] = np.nan
            return sims.reshape(len(X), 2, 2) if args.baseline else sims
        problem = ModelProblem(nInput=1, lb=-5, ub=5, obs=obs.reshape(2, 2) if args.baseline else obs,
                               mask=mask.reshape(2, 2) if args.baseline and masked else mask, simFunc=simulate)
        method = cls(maxIters=2, seed=17) if cls is IES else cls()
        options = {'r': np.eye(3 if masked else 4)} if cls in (ES, IES) else (
            {'threshold': 100} if cls is GLUE else {'eliteSize': 8, 'seed': 17})
        result = method.run(problem, np.linspace(-1., 1., 24)[:, None], **options)
        for field in ('samples', 'simulations', 'scores', 'bestDecs', 'bestSim', 'weights'):
            value = getattr(result, field)
            if value is not None:
                arrays[f'{cls.__name__}_{masked}_{field}'] = value
path = Path('agent/verification/1004-observation-migration-baseline.npz')
if args.baseline:
    np.savez(path, **arrays)
    print(f'Saved {len(arrays)} pre-migration arrays')
else:
    baseline = np.load(path)
    assert set(baseline.files) == set(arrays)
    for name, value in arrays.items():
        np.testing.assert_array_equal(value, baseline[name], err_msg=name)
    print(f'{len(arrays)} arrays exactly unchanged; 4 algorithms x 2 mask cases')
