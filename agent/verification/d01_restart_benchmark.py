"""Compare Boxmin start policies independently of production restart loops."""

import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from d01_optimizer_benchmark import CASE_SPECS, makeData, makeModel, Boxmin


def main():
    rows = []
    for profile in ['default', 'scaled_ard']:
        for family in ['GPR', 'KRG']:
            for case in CASE_SPECS:
                rawX, rawY, testX, testY = makeData(case)
                for seed in range(10):
                    model = makeModel(family, profile, 'Boxmin', seed)
                    trainX, trainY = model.prepareTrainingData(rawX, rawY)
                    infos, upper, lower = model.setting.getParaInfos(model.getParaList())
                    upper, lower = upper.ravel(), lower.ravel()
                    initial = np.empty(lower.size)
                    for name, indices in infos.items():
                        value = model.setting.get(name)
                        initial[indices] = np.log(value) if model.setting.parLog[name] else value
                    initial = np.clip(initial, lower, upper)
                    if family == 'KRG':
                        trend, distances = model._initialize(trainX)

                    def objective(point):
                        model.setting.setVals(infos, point)
                        if family == 'GPR':
                            return model._objfunc(trainX, trainY, record=False)
                        return model._objFunc(trainY, trend, distances, record=False)

                    problem = SimpleNamespace(lb=lower, ub=upper, nInput=lower.size, objFunc=objective)
                    rng = np.random.default_rng(seed)
                    starts = [initial, *rng.uniform(lower, upper, (9, lower.size))]
                    candidates = []
                    for start in starts:
                        optimizer = Boxmin()
                        before = time.perf_counter()
                        point, value = optimizer.run(problem, xInit=start)
                        elapsed = time.perf_counter() - before
                        checked = float(objective(point))
                        np.testing.assert_allclose(checked, value, rtol=1e-10, atol=1e-10)
                        assert np.isfinite(checked) and np.all(point >= lower) and np.all(point <= upper)
                        model.fitModel(trainX, trainY)
                        prediction = model.predict(testX)
                        assert np.isfinite(prediction).all()
                        error = float(np.sqrt(np.mean(((prediction-testY)/np.std(testY, axis=0))**2)))
                        candidates.append(dict(objective=checked, normalized_rmse=error,
                                               evaluations=optimizer.nv, seconds=elapsed,
                                               start=start.tolist(), point=np.ravel(point).tolist()))
                    policies = {}
                    for policy, indices in [('configured_first', list(range(9))),
                                            ('all_random', list(range(1, 10)))]:
                        results = []
                        for count in [1, 2, 3, 5, 9]:
                            subset = [candidates[index] for index in indices[:count]]
                            selected = min(subset, key=lambda item: item['objective'])
                            results.append(dict(total_runs=count, objective=selected['objective'],
                                                normalized_rmse=selected['normalized_rmse'],
                                                evaluations=sum(item['evaluations'] for item in subset),
                                                seconds=sum(item['seconds'] for item in subset)))
                        assert all(a['objective'] >= b['objective'] for a, b in zip(results, results[1:]))
                        policies[policy] = results
                    rows.append(dict(profile=profile, family=family, case=case, seed=seed,
                                     candidates=candidates, policies=policies))
                print(profile, family, case, flush=True)
    output = Path(__file__).with_name('0918-d01-restart-results.json')
    output.write_text(json.dumps(dict(seeds=list(range(10)), results=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
