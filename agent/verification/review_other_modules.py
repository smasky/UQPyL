"""单模型单输出约定下的 OM01–OM04 独立复核，历史证据另存。"""

from itertools import combinations
import json
from pathlib import Path

import numpy as np

from UQPyL.calibration import util
from UQPyL.calibration.methods._ensemble import anomalyGain, ensembleGain
from UQPyL.inference import AMH, MH, MH_Gibbs
from UQPyL.optimization.core.non_dominated_sort import NDSort
from UQPyL.optimization.metric import GD, HV, IGD
from UQPyL.optimization.moea import NSGAII
from UQPyL.optimization.soea import GA, PSO
from UQPyL.problem import ModelProblem, Problem, Space
from UQPyL.surrogate.base import MultiSurrogate
from UQPyL.surrogate.gp.gaussian_process import GPR
from UQPyL.surrogate.gp.kernel.rbf_kernel import RBF
from UQPyL.surrogate.regression import LinearRegression, PolynomialRegression
from UQPyL.surrogate.scaler import StandardScaler


records = []


def record(group, case, discrepancy=False, **details):
    records.append({"group": group, "case": case, "discrepancy": bool(discrepancy), **details})


def checkClose(actual, expected, atol=2e-11):
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=atol)
    return float(np.max(np.abs(np.asarray(actual) - expected), initial=0))


def reviewMetrics():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    sim = np.array([[1.1, 1.9, 3.2, 3.8], [1.2, 2.2, 3.2, 4.2]])
    centeredObs = obs - obs.mean()
    centeredSim = sim - sim.mean(axis=1, keepdims=True)
    correlation = (centeredSim @ centeredObs) / (np.linalg.norm(centeredSim, axis=1) * np.linalg.norm(centeredObs))
    expectedNse = 1 - np.sum((sim - obs) ** 2, axis=1) / np.sum(centeredObs**2)
    expectedKge = 1 - np.sqrt(
        (correlation - 1) ** 2 + (sim.std(axis=1) / obs.std() - 1) ** 2 + (sim.mean(axis=1) / obs.mean() - 1) ** 2
    )
    references = {
        "nse": expectedNse,
        "r2": expectedNse,
        "pbias": 100 * np.sum(sim - obs, axis=1) / obs.sum(),
        "pearson_r": correlation,
        "kge": expectedKge,
        "rfactor": 0.4 / obs.std(),
    }
    for scale in [1.0, 1e-3, 1e-5, 1e-9, 1e-200, 1e200]:
        for name, reference in references.items():
            args = (
                (obs * scale, (obs - 0.2) * scale, (obs + 0.2) * scale)
                if name == "rfactor"
                else (obs * scale, sim * scale)
            )
            try:
                actual = getattr(util, name)(*args)
            except ValueError as error:
                record("OM01", f"{name}/scale={scale}", True, error=str(error), expected=np.asarray(reference).tolist())
            else:
                record(
                    "OM01",
                    f"{name}/scale={scale}",
                    max_error=checkClose(actual, reference),
                    actual=np.asarray(actual).tolist(),
                )
        # 极端尺度仅用于此次修复的无量纲指标；有量纲损失保留原普通范围对照。
        if scale in (1e-200, 1e200):
            continue
        for name, reference in [
            ("mse", np.mean((sim - obs) ** 2, axis=1) * scale**2),
            ("mae", np.mean(np.abs(sim - obs), axis=1) * scale),
            ("rmse", np.sqrt(np.mean((sim - obs) ** 2, axis=1)) * scale),
        ]:
            actual = getattr(util, name)(obs * scale, sim * scale)
            record("calibration_controls", f"{name}/scale={scale}", max_error=checkClose(actual, reference, atol=0))


def reviewProposals():
    nChains, gamma, seed = 256, 0.1, 641
    for method in [MH, AMH, MH_Gibbs]:
        for distribution in ["gauss", "uniform"]:
            unitSteps = []
            for span in [0.1, 1.0, 10.0]:
                algorithm = method(nChains=nChains, warmUp=0, maxIters=1, saveFlag=False, verboseFlag=False)
                algorithm.rng = np.random.default_rng(seed)
                lb, ub = np.array([[0.0]]), np.array([[span]])
                current = np.full((nChains, 1), span / 2)
                # 与 run 中的初始提议协方差构造一致。
                covariance = [np.diag([(gamma * span) ** 2]) for _ in range(nChains)]
                if method is MH_Gibbs:
                    proposed = algorithm.f_prop(0, current, distribution, covariance, ub, lb)
                else:
                    proposed = algorithm.f_prop(current, distribution, covariance, ub, lb)
                referenceRng = np.random.default_rng(seed)
                if distribution == "gauss":
                    reference = referenceRng.normal(0, gamma * span, (nChains, 1))
                else:
                    reference = referenceRng.uniform(-gamma * span, gamma * span, (nChains, 1))
                # 这些对照均不触及边界，反射不会影响步长比较。
                assert np.all((current + reference > lb) & (current + reference < ub))
                actualStep = (proposed - current) / span
                unitSteps.append(actualStep)
                error = float(np.max(np.abs(actualStep - reference / span)))
                wasAffected = method is MH_Gibbs or distribution == "uniform"
                checkClose(actualStep, reference / span)
                record(
                    "OM02" if wasAffected else "inference_controls",
                    f"{method.__name__}/{distribution}/span={span}",
                    error > 1e-12,
                    max_error=error,
                    step_rms=float(np.sqrt(np.mean(actualStep**2))),
                    expected_step_rms=float(np.sqrt(np.mean((reference / span) ** 2))),
                )
            checkClose(unitSteps[1], unitSteps[0])
            checkClose(unitSteps[2], unitSteps[0])
            # 公共 run 入口：同一个均匀目标，仅更换输入单位。
            runSamples = []
            for span in [1.0, 10.0]:
                problem = Problem(nInput=1, nObj=1, lb=0, ub=span, objFunc=lambda x: np.zeros((len(x), 1)))
                algorithm = method(
                    nChains=8,
                    warmUp=3,
                    maxIters=12,
                    propDist=distribution,
                    saveFlag=False,
                    logFlag=False,
                    verboseFlag=False,
                )
                result = algorithm.run(problem, gamma=gamma, seed=seed)
                assert result.decs.shape == (8, 12, 1)
                assert result.FEs == 120
                assert np.all(result.accepted)
                runSamples.append(result.decs / span)
            runError = float(np.max(np.abs(runSamples[0] - runSamples[1])))
            wasAffected = method is MH_Gibbs or distribution == "uniform"
            checkClose(runSamples[0], runSamples[1])
            record(
                "OM02" if wasAffected else "inference_controls",
                f"run/{method.__name__}/{distribution}/unit_change",
                runError > 1e-12,
                max_unit_sample_difference=runError,
                stored_draws=12,
                proposal_updates=11,
                warm_up=3,
                target="uniform",
            )


def regressionReference(trainX, trainY, testX, degree, loss):
    features = np.hstack([trainX**exponent for exponent in range(1, degree + 1)])
    testFeatures = np.hstack([testX**exponent for exponent in range(1, degree + 1)])
    if loss == "Origin":
        augmented = np.c_[features, np.ones(len(features))]
        coefficient = np.linalg.lstsq(augmented, trainY, rcond=None)[0]
        return np.c_[testFeatures, np.ones(len(testFeatures))] @ coefficient
    xMean, yMean = features.mean(axis=0), trainY.mean(axis=0)
    centered = features - xMean
    coefficient = np.linalg.solve(centered.T @ centered + 0.1 * np.eye(degree), centered.T @ (trainY - yMean))
    return (testFeatures - xMean) @ coefficient + yMean


def reviewRegression():
    trainX = np.linspace(-1, 1, 21).reshape(-1, 1)
    testX = np.array([[-0.7], [0.2], [0.8]])
    for modelClass, degree in [(LinearRegression, 1), (PolynomialRegression, 2)]:
        trainY = np.hstack((1 + 2 * trainX, 3 - trainX))
        if degree == 2:
            trainY += np.hstack((0.5 * trainX**2, 0.75 * trainX**2))
        for loss in ["Origin", "Ridge"]:
            expected = regressionReference(trainX, trainY, testX, degree, loss)
            for scaled in [False, True]:
                kwargs = {"lossType": loss, "C": 0.1, "C_attr": None}
                if degree == 2:
                    kwargs.update(degree=2, degree_attr=None)

                def newModel():
                    return modelClass(scalers=(None, StandardScaler() if scaled else None), **kwargs)

                model = newModel()
                try:
                    model.fit(trainX, trainY)
                except ValueError as error:
                    assert "single output" in str(error) and "MultiSurrogate" in str(error)
                else:
                    raise AssertionError("A single surrogate accepted multiple output columns")
                multi = MultiSurrogate(2, [newModel(), newModel()]).fit(trainX, trainY)
                actual = multi.predict(testX)
                assert actual.shape == expected.shape
                record(
                    "OM03",
                    f"{modelClass.__name__}/{loss}/scaled={scaled}",
                    expected_shape=list(expected.shape),
                    actual_shape=list(actual.shape),
                    max_error=checkClose(actual, expected),
                    fit_accepted=False,
                    prediction_scope="MultiSurrogate",
                )
                single = newModel().fit(trainX, trainY[:, :1]).predict(testX)
                record(
                    "regression_controls",
                    f"single/{modelClass.__name__}/{loss}/scaled={scaled}",
                    max_error=checkClose(single, expected[:, :1]),
                )
                record(
                    "regression_controls",
                    f"container/{modelClass.__name__}/{loss}/scaled={scaled}",
                    max_error=checkClose(multi.predict(testX), expected),
                )


def reviewGpr():
    trainX = np.linspace(-1, 1, 7).reshape(-1, 1)
    testX = np.array([[-0.85], [-0.1], [0.75]])
    outputs = np.hstack((np.sin(2 * trainX), 3 + np.cos(trainX)))
    noise, length = 0.03, 0.7
    for nOutput in [1, 2]:
        for scaled in [False, True]:
            children = [
                GPR(
                    kernel=RBF(length_scale=length, length_attr=None),
                    C=noise,
                    C_attr=None,
                    scalers=(StandardScaler() if scaled else None, StandardScaler() if scaled else None),
                )
                for _ in range(nOutput)
            ]
            model = children[0] if nOutput == 1 else MultiSurrogate(nOutput, children)
            trainY = outputs[:, :nOutput]
            model.fit(trainX, trainY)
            xScale = trainX.std(axis=0, ddof=1) if scaled else np.ones(1)
            xMean = trainX.mean(axis=0) if scaled else np.zeros(1)
            yScale = trainY.std(axis=0, ddof=1) if scaled else np.ones(nOutput)
            yMean = trainY.mean(axis=0) if scaled else np.zeros(nOutput)
            x, t, y = (trainX - xMean) / xScale, (testX - xMean) / xScale, (trainY - yMean) / yScale

            def kernel(a, b):
                return np.exp(-0.5 * np.sum((a[:, None] - b[None, :]) ** 2, axis=2) / length**2)

            covariance = kernel(x, x) + noise * np.eye(len(x))
            cross = kernel(t, x)
            solvedY = np.linalg.solve(covariance, y)
            mean = cross @ solvedY * yScale + yMean
            variance = (1 - np.diag(cross @ np.linalg.solve(covariance, cross.T)))[:, None] * yScale**2
            actualMean, actualVariance = model.predict(testX, returnVar=True)
            likelihood = (
                0.5 * np.sum(y * solvedY)
                + 0.5 * nOutput * np.linalg.slogdet(covariance)[1]
                + 0.5 * len(x) * nOutput * np.log(2 * np.pi)
            )
            record(
                "gpr_controls",
                f"outputs={nOutput}/scaled={scaled}",
                mean_error=checkClose(actualMean, mean),
                variance_error=checkClose(actualVariance, variance),
                likelihood_error=checkClose(sum(child.fitState["objective"] for child in children), likelihood),
            )


def unionVolume(points, reference):
    points = points[np.all(points <= reference, axis=1)]
    volume = 0.0
    for count in range(1, len(points) + 1):
        for indices in combinations(range(len(points)), count):
            lower = np.max(points[list(indices)], axis=0)
            volume += (-1) ** (count + 1) * np.prod(np.maximum(reference - lower, 0))
    return volume


def independentFronts(points):
    remaining = list(range(len(points)))
    ranks = np.zeros(len(points))
    rank = 0
    while remaining:
        rank += 1
        front = [
            i
            for i in remaining
            if not any(np.all(points[j] <= points[i]) and np.any(points[j] < points[i]) for j in remaining)
        ]
        ranks[front] = rank
        remaining = [i for i in remaining if i not in front]
    return ranks


def reviewOptimization():
    for dimension in [2, 3]:
        for seed in range(3):
            rng = np.random.default_rng(seed)
            points = rng.uniform(0.1, 0.7, (4, dimension))
            reference = np.ones(dimension)
            variants = [
                points,
                points[::-1],
                np.vstack((points, points[0])),
                np.vstack((points, np.full(dimension, 0.9))),
                np.vstack((points, np.full(dimension, 1.2))),
            ]
            for index, variant in enumerate(variants):
                record(
                    "optimization_controls",
                    f"HV/{dimension}D/seed={seed}/variant={index}",
                    max_error=checkClose(HV(variant, reference, normalize=False), unionVolume(variant, reference)),
                )
            optimum = rng.uniform(0, 1, (5, dimension))
            distance = np.sqrt(np.sum((points[:, None] - optimum[None, :]) ** 2, axis=2))
            record(
                "optimization_controls",
                f"GD_IGD/{dimension}D/seed={seed}",
                gd_error=checkClose(GD(points, optimum), distance.min(axis=1).mean()),
                igd_error=checkClose(IGD(points, optimum), distance.min(axis=0).mean()),
            )
            points = rng.integers(0, 5, (12, dimension)).astype(float)
            actual, _ = NDSort(points)
            record(
                "optimization_controls",
                f"NDSort/{dimension}D/seed={seed}",
                max_error=checkClose(actual, independentFronts(points)),
            )
    for method in [PSO, GA, NSGAII]:
        for seed in [42, 43]:
            count = [0]

            def objective(x):
                count[0] += len(x)
                first = np.sum(x**2, axis=1, keepdims=True)
                return np.hstack((first, np.sum((x - 0.5) ** 2, axis=1, keepdims=True))) if method is NSGAII else first

            def constraint(x):
                return np.sum(x, axis=1, keepdims=True) - 0.8

            problem = Problem(
                nInput=2, nObj=2 if method is NSGAII else 1, lb=-1, ub=1, nCon=1, objFunc=objective, conFunc=constraint
            )
            kwargs = dict(nPop=16, maxIters=4, maxFEs=160, saveFlag=False, logFlag=False, verboseFlag=False)
            if method is NSGAII:
                kwargs["hvFlag"] = False
            result = method(**kwargs).run(problem, seed=seed)
            assert result.FEs == count[0]
            assert result.bestFeasible and np.all((result.bestDecs >= -1) & (result.bestDecs <= 1))
            assert np.all(constraint(result.bestDecs) <= 0)
            expected = objective(result.bestDecs)
            record(
                "optimization_controls",
                f"run/{method.__name__}/seed={seed}",
                objective_error=checkClose(result.bestObjs, expected),
                constraint_error=checkClose(result.bestCons, constraint(result.bestDecs)),
                fes=result.FEs,
            )


def reviewGain():
    rng = np.random.default_rng(63)
    x = rng.normal(size=(6, 2))
    x -= x.mean(axis=0)
    for nObs in [3, 10]:
        y = rng.normal(size=(6, nObs))
        y -= y.mean(axis=0)
        for scale in [1.0, 1e-5]:
            for noise in [0.0, 0.2]:
                scaledY = y * scale
                r = noise * scale**2 * np.eye(nObs)
                cross = x.T @ scaledY / 5
                covariance = scaledY.T @ scaledY / 5
                expected = cross @ np.linalg.pinv(covariance + r, rcond=nObs * np.finfo(float).eps)
                gain, info = ensembleGain(cross, covariance, r)
                anomaly, anomalyInfo = anomalyGain(x, scaledY, None if noise == 0 else r)
                record(
                    "gain_controls",
                    f"nObs={nObs}/scale={scale}/noise={noise}",
                    gain_error=checkClose(gain, expected, atol=1e-7),
                    anomaly_error=checkClose(anomaly, expected, atol=1e-7),
                    solver=info["solver"],
                    anomaly_solver=anomalyInfo["solver"],
                )


def reviewProblem():
    space = Space(1, lb=0, ub=2, varType=[2], varSet={0: [0.25, 0.75]})
    for helper in ["map_discrete_vars", "apply_var_type", "transform"]:
        for integer in [True, False]:
            inputs = np.array([[0], [1], [2]], dtype=int if integer else float)
            before = inputs.copy()
            actual = getattr(space, helper)(inputs)
            expected = np.array([[0.25], [0.75], [0.75]])
            np.testing.assert_array_equal(inputs, before)
            checkClose(actual, expected)
            record(
                "OM04" if integer else "problem_controls",
                f"{helper}/integer={integer}",
                actual=actual.tolist(),
                expected=expected.tolist(),
            )
    mixed = Space(3, lb=[0, 0, 0], ub=[1, 3, 2], varType=[0, 1, 2], varSet={2: [0.25, 0.75]})
    real = np.array([[0.1, 0, 0.25], [0.5, 2, 0.75], [1, 3, 0.75]])
    record(
        "problem_controls",
        "mixed_roundtrip",
        max_error=checkClose(mixed.unit_to_space(mixed.space_to_unit(real)), real),
    )
    points = np.array([[0.2, 0.4], [0.7, 0.1]])
    for isModel in [False, True]:
        for target in [None, "objs", "cons"]:
            calls = []

            def objective(x, *context):
                calls.append("objs")
                if context:
                    checkClose(context[0].sims, x**2)
                return np.sum(x**2, axis=1, keepdims=True)

            def constraint(x, *context):
                calls.append("cons")
                return x[:, :1] - 0.5

            kwargs = dict(nInput=2, nObj=1, nCon=1, lb=0, ub=1, objFunc=objective, conFunc=constraint)
            problem = ModelProblem(simFunc=lambda x: x**2, **kwargs) if isModel else Problem(**kwargs)
            result = problem.evaluate(points, target=target)
            assert calls == (["objs", "cons"] if target is None else [target])
            if result.objs is not None:
                checkClose(result.objs, np.sum(points**2, axis=1, keepdims=True))
            if result.cons is not None:
                checkClose(result.cons, points[:, :1] - 0.5)
            record("problem_controls", f"evaluate/model={isModel}/target={target}")


def main():
    for review in [
        reviewMetrics,
        reviewProposals,
        reviewRegression,
        reviewGpr,
        reviewOptimization,
        reviewGain,
        reviewProblem,
    ]:
        review()
    assert not any(row["discrepancy"] for row in records)
    summary = {
        "records": len(records),
        "discrepancies": sum(row["discrepancy"] for row in records),
        "controls": sum(not row["group"].startswith("OM") for row in records),
    }
    payload = {"summary": summary, "records": records}
    destination = Path(__file__).with_name("1002-other-modules-single-output.json")
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for row in records:
        if row["discrepancy"]:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
