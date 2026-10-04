"""Repeatable work counts and trajectory fingerprints for C13/C14."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np

from UQPyL.inference import MH, MH_Gibbs, AMH, DEMC, DREAM_ZS
from UQPyL.optimization.moea import NSGAII
from UQPyL.problem import Problem


def fingerprint(result):
    fields = {}
    for name in ('decs', 'objs', 'cons', 'logProb', 'accepted', 'feasibleMask',
                 'acceptanceRate', 'bestDecs', 'bestObjs', 'bestCons'):
        value = getattr(result, name, None)
        fields[name] = None if value is None else hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
    return fields


def inferenceCase(methodClass, draws, save=False):
    options = {'maxIterTimes' if methodClass in (AMH, DEMC) else 'maxIters': draws}
    method = methodClass(nChains=4, warmUp=2, verboseFlag=False, logFlag=False,
                         saveFlag=save, saveFreq=10, **options)
    problem = Problem(nInput=2, nObj=1, lb=0., ub=1., objFunc=lambda x: np.sum(x*x, axis=1)[:, None])
    originalCollect = method.state._collect
    originalDecode = method._decodeDecs
    originalBuild = method.state.buildResult
    counts = dict(decoded_history_rows=0, full_result_builds=0, result_rows_copied=0)
    collecting = False

    def collect(*args):
        nonlocal collecting
        collecting = True
        try:
            return originalCollect(*args)
        finally:
            collecting = False

    def decode(x):
        if collecting:
            counts['decoded_history_rows'] += len(x)
        return originalDecode(x)

    def build(*args, **kwargs):
        result = originalBuild(*args, **kwargs)
        counts['full_result_builds'] += 1
        counts['result_rows_copied'] += result.decs.shape[0]*result.decs.shape[1]
        return result

    with TemporaryDirectory() as directory:
        problem.workDir = directory
        with patch.object(method.state, '_collect', collect), patch.object(method, '_decodeDecs', decode), \
                patch.object(method.state, 'buildResult', build):
            start = perf_counter()
            result = method.run(problem, seed=17)
            elapsed = perf_counter()-start
    return dict(method=method.name, draws=draws, save=save, seconds=elapsed, **counts,
                fingerprint=fingerprint(result), fes=result.FEs, iters=result.iters,
                mean_log_prob=result.history.meanLogProbHistory,
                acceptance=result.history.acceptanceRateHistory,
                feasible=result.history.feasibleRateHistory, best=result.history.bestObjHistory)


def optimizationCase():
    module = importlib.import_module('UQPyL.optimization.runtime.result')
    original = module.HV
    calls = []
    def hv(*args, **kwargs):
        start = perf_counter()
        value = original(*args, **kwargs)
        calls.append(dict(points=len(args[0]), samples=kwargs.get('nSamples', 1_000_000),
                          seconds=perf_counter()-start))
        return value
    problem = Problem(nInput=2, nObj=4, lb=0., ub=1.,
                      objFunc=lambda x: np.column_stack([x[:, 0], 1-x[:, 0], x[:, 1], 1-x[:, 1]]))
    with patch.object(module, 'HV', hv):
        start = perf_counter()
        result = NSGAII(nPop=12, maxIters=2, verboseFlag=False, logFlag=False,
                        saveFlag=False, historyFreq=1).run(problem, seed=1)
        elapsed = perf_counter()-start
    return dict(seconds=elapsed, calls=calls, fes=result.FEs, fingerprint=fingerprint(result),
                populations=[hashlib.sha256(row['decs'].tobytes()).hexdigest()
                             for row in result.history.populations], metrics=result.history.metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = dict(inference=[inferenceCase(MH, draws, save) for save in (False, True)
                             for draws in (40, 80, 160)],
                  methods=[inferenceCase(cls, 40) for cls in (MH_Gibbs, AMH, DEMC, DREAM_ZS)],
                  optimization=optimizationCase())
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(args.output)
