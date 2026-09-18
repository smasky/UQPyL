"""Paired GPR/KRG optimizer audit; run with BLAS thread counts set to one."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import scipy

from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Guass
from UQPyL.surrogate.scaler import MinMaxScaler, StandardScaler
from UQPyL.surrogate.util.boxmin import Boxmin
from UQPyL.surrogate.util.lbfgsb import LBFGSB
import UQPyL.surrogate.util.lbfgsb as lbfgsbModule


CASE_SPECS = {
    'smooth_1d': (24, 1), 'noisy_1d': (32, 1),
    'anisotropic_2d': (40, 2), 'multioutput_2d': (40, 2),
    'interaction_5d': (60, 5), 'physical_units_2d': (40, 2),
}


def makeData(case):
    count, dimension = CASE_SPECS[case]
    rng = np.random.default_rng(20260918 + list(CASE_SPECS).index(case))
    trainU = rng.uniform(0., 1., (count, dimension))
    testU = rng.uniform(0., 1., (256, dimension))
    if dimension == 1:
        trainU = np.linspace(0., 1., count)[:, None]
        testU = np.linspace(0., 1., 256)[:, None]

    def truth(values):
        if case in ('smooth_1d', 'noisy_1d'):
            return np.sin(6. * values[:, :1]) + .2 * values[:, :1]
        if case == 'anisotropic_2d':
            return (np.sin(12. * values[:, 0]) + .3 * np.cos(2. * values[:, 1]))[:, None]
        if case == 'multioutput_2d':
            return np.column_stack([
                np.sin(6. * values[:, 0]) + .2 * values[:, 1],
                np.cos(4. * values[:, 1]) + .3 * values[:, 0],
            ])
        if case == 'interaction_5d':
            return (np.sin(5. * values[:, 0]) + .5 * values[:, 1] ** 2
                    + values[:, 2] * values[:, 3] + .1 * values[:, 4])[:, None]
        return (100. + 20. * np.sin(6. * values[:, 0]) + 5. * values[:, 1])[:, None]

    trainY, testY = truth(trainU), truth(testU)
    if case == 'noisy_1d':
        trainY += rng.normal(0., .05, trainY.shape)
    if case == 'physical_units_2d':
        trainX, testX = [np.array([-500., 10.]) + values * np.array([1000., .02])
                        for values in (trainU, testU)]
    else:
        trainX, testX = trainU, testU
    return trainX, trainY, testX, testY


class AuditedOptimizer:
    type = 'MP'

    def __init__(self, name):
        self.name = name
        self.optimizer = Boxmin() if name == 'Boxmin' else LBFGSB()
        self.runs = []

    def run(self, problem, xInit=None, seed=None):
        lower, upper = np.ravel(problem.lb), np.ravel(problem.ub)
        record = {'seed': int(seed), 'evaluations': 0, 'nonfinite_objectives': 0,
                  'out_of_bounds': 0, 'first_point': None}
        self.runs.append(record)

        def countedObjective(point):
            point = np.ravel(point)
            if record['first_point'] is None:
                record['first_point'] = point.tolist()
            record['evaluations'] += 1
            record['out_of_bounds'] += int(np.any(point < lower) or np.any(point > upper))
            value = problem.objFunc(point)
            record['nonfinite_objectives'] += int(not np.all(np.isfinite(value)))
            return value

        countedProblem = SimpleNamespace(lb=problem.lb, ub=problem.ub,
                                         nInput=problem.nInput, objFunc=countedObjective)
        if self.name == 'LBFGSB':
            originalMinimize = lbfgsbModule.minimize

            def captureMinimize(*args, **kwargs):
                result = originalMinimize(*args, **kwargs)
                record.update(success=bool(result.success), status=int(result.status),
                              message=str(result.message), iterations=int(result.nit),
                              scipy_nfev=int(result.nfev))
                return result

            with patch.object(lbfgsbModule, 'minimize', captureMinimize):
                best, objective = self.optimizer.run(countedProblem, xInit=xInit, seed=seed)
        else:
            best, objective = self.optimizer.run(countedProblem, xInit=xInit, seed=seed)
        record['returned_point'] = np.ravel(best).tolist()
        record['returned_objective'] = float(objective)
        assert np.all(np.isfinite(best)) and np.all(best >= lower) and np.all(best <= upper)
        assert record['out_of_bounds'] == 0
        return best, objective


def makeModel(family, profile, optimizer, seed):
    options = {'optimizer': optimizer}
    if profile == 'scaled_ard':
        options['scalers'] = (MinMaxScaler(), StandardScaler())
        if family == 'GPR':
            options.update(
                kernel=RBF(heterogeneous=True, length_scale=.3,
                           length_attr={'lb': .03, 'ub': 5., 'type': 'float', 'log': True}),
                C=1e-6, C_attr={'lb': 1e-10, 'ub': .1, 'type': 'float', 'log': True},
            )
        else:
            options['kernel'] = Guass(heterogeneous=True, theta=1.,
                                     theta_attr={'lb': .02, 'ub': 1. / (2. * .03**2),
                                                 'type': 'float', 'log': True})
    model = (GPR if family == 'GPR' else KRG)(**options)
    model.rng = np.random.default_rng(seed)
    return model


def runFit(family, profile, case, optimizerName, seed, timingRepeats):
    trainX, trainY, testX, testY = makeData(case)
    audit = AuditedOptimizer(optimizerName)
    model = makeModel(family, profile, audit, seed)
    record = dict(family=family, profile=profile, case=case, optimizer=optimizerName,
                  seed=seed, n_train=len(trainX), n_input=trainX.shape[1], n_output=trainY.shape[1],
                  fit_success=False, failure=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            model.fit(trainX, trainY)
            prediction, variance = model.predict(testX, returnVar=True)
            assert np.isfinite(prediction).all() and np.isfinite(variance).all()
            assert (variance >= 0).all()
            objective = float(model.fitState['objective'])
            assert np.isfinite(objective)
            rmse = np.sqrt(np.mean((prediction-testY)**2, axis=0))
            normalizedRmse = np.sqrt(np.mean(((prediction-testY)/np.std(testY, axis=0))**2))
            record.update(fit_success=True, objective=objective, rmse_per_output=rmse.tolist(),
                          normalized_rmse=float(normalizedRmse),
                          parameters={name: np.atleast_1d(model.setting.get(name)).tolist()
                                      for name in model.getParaList()},
                          timings_seconds=[])
            # Time the production implementation without the audit wrapper.
            for _ in range(timingRepeats):
                timedModel = makeModel(family, profile, optimizerName, seed)
                start = time.perf_counter()
                timedModel.fit(trainX, trainY)
                record['timings_seconds'].append(time.perf_counter()-start)
                np.testing.assert_allclose(timedModel.fitState['objective'], objective, rtol=1e-12, atol=1e-12)
            record['median_fit_seconds'] = float(np.median(record['timings_seconds']))
        except Exception as error:
            record['fit_success'] = False
            record['failure'] = f'{type(error).__name__}: {error}'
        record['warnings'] = sorted(set(str(item.message) for item in caught))
    record['optimizer_runs'] = audit.runs
    record['objective_evaluations'] = sum(run['evaluations'] for run in audit.runs)
    record['nonfinite_objectives'] = sum(run['nonfinite_objectives'] for run in audit.runs)
    record['out_of_bounds'] = sum(run['out_of_bounds'] for run in audit.runs)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--cases', nargs='+', choices=list(CASE_SPECS), default=list(CASE_SPECS))
    parser.add_argument('--profiles', nargs='+', choices=['default', 'scaled_ard'], default=['default', 'scaled_ard'])
    parser.add_argument('--timing-repeats', type=int, default=2)
    args = parser.parse_args()
    report = {'metadata': {'python': platform.python_version(), 'numpy': np.__version__,
                          'scipy': scipy.__version__, 'platform': platform.platform(),
                          'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                          'thread_env': {name: os.environ.get(name) for name in
                                         ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
                          'timing_repeats': args.timing_repeats}, 'datasets': {}, 'results': []}
    for case in args.cases:
        data = makeData(case)
        report['datasets'][case] = {'sha256': hashlib.sha256(b''.join(a.tobytes() for a in data)).hexdigest(),
                                    'shapes': [list(a.shape) for a in data]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Warm both implementations before collecting timing evidence.
    for family in ['GPR', 'KRG']:
        for name in ['Boxmin', 'LBFGSB']:
            warm = makeModel(family, 'default', name, 999)
            warm.fit(*makeData('smooth_1d')[:2])
    for profile in args.profiles:
        for family in ['GPR', 'KRG']:
            for case in args.cases:
                for seed in args.seeds:
                    pair = []
                    order = ['Boxmin', 'LBFGSB'] if seed % 2 == 0 else ['LBFGSB', 'Boxmin']
                    for optimizer in order:
                        row = runFit(family, profile, case, optimizer, seed, args.timing_repeats)
                        pair.append(row)
                        report['results'].append(row)
                    if all(row['fit_success'] for row in pair):
                        np.testing.assert_allclose(
                            [r['first_point'] for r in pair[0]['optimizer_runs']],
                            [r['first_point'] for r in pair[1]['optimizer_runs']],
                            rtol=1e-14, atol=1e-13,  # Boxmin's affine round trip.
                        )
                        assert [r['seed'] for r in pair[0]['optimizer_runs']] == [
                            r['seed'] for r in pair[1]['optimizer_runs']]
                    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                print(f'{profile} {family} {case}: {len(report["results"])} fits recorded', flush=True)


if __name__ == '__main__':
    main()
