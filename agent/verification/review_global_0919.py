"""Read-only runtime probes for the three-pass project review."""
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
from UQPyL.problem import Problem, ModelProblem
from UQPyL.inference import MH
from UQPyL.optimization.soea import GA
from UQPyL.optimization.moea import NSGAII
from UQPyL.optimization.expensive import MOASMO
from UQPyL.analysis import Sobol, Morris
from UQPyL.doe import SaltelliDesign, MorrisDesign
from UQPyL.calibration import ES, IES
from UQPyL.surrogate import MultiSurrogate
from UQPyL.surrogate.mars import MARS
from UQPyL.surrogate.regression.linear_regression import LinearRegression

QUIET = dict(verboseFlag=False, logFlag=False, saveFlag=False)
results = {}
def probe(name, func):
    try:
        results[name] = func()
    except Exception as error:
        results[name] = {"probe_error": type(error).__name__, "message": str(error)}
def problem(nObj=1, optType="min"):
    return Problem(nInput=2, nObj=nObj, lb=0., ub=1., optType=optType,
                   objFunc=lambda x: np.column_stack([np.sum((x-i*.2)**2, axis=1) for i in range(nObj)]))

def inferenceHistory():
    method = MH(nChains=2, warmUp=0, maxIters=8, **QUIET)
    first = method.run(problem(), seed=1)
    before = [row.copy() for row in first.history.iterToFEs]
    method.maxIters = 3
    second = method.run(problem(), seed=2)
    return dict(first_draws=first.decs.shape[1], original_history=len(before),
                history_after_second_run=len(first.history.iterToFEs),
                history_shared=first.history is second.history)
probe("inference_result_history_shared", inferenceHistory)

def inferenceSigns():
    p = problem(optType="max")
    result = MH(nChains=1, warmUp=0, maxIters=6, **QUIET).run(p, seed=1)
    actual = p.evaluate(result.decs.reshape(-1, 2)).objs
    return dict(returned=result.objs.ravel().tolist(), actual=actual.ravel().tolist(),
                best=result.bestObjs.ravel().tolist())
probe("inference_objective_units", inferenceSigns)

def inferenceWork():
    out = []
    for draws in [40, 80, 160]:
        method = MH(nChains=2, warmUp=0, maxIters=draws, **QUIET)
        original = method._decodeDecs
        rows = []
        def decode(values):
            array = np.asarray(values)
            if array.ndim == 2:
                rows.append(len(array))
            return original(values)
        method._decodeDecs = decode
        method.run(problem(), seed=1)
        # Evaluation batches are 2 rows. History collection grows beyond 2.
        out.append(dict(draws=draws, history_rows=sum(r for r in rows if r>2)))
    return out
probe("inference_repeated_history_processing", inferenceWork)

def unitMeta():
    p = Problem(nInput=2, nObj=1, lb=[0.,0.], ub=[100.,1.], objFunc=lambda x:x.sum(axis=1)[:,None])
    out = {}
    for output in ["real", "unit"]:
        x, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(p, 512, seed=2, output=output)
        r = Sobol(**QUIET).analyze(p, x, meta=meta)
        out[output] = r.getMetric("S1").values.tolist()
    return out
probe("doe_analysis_unit_metadata", unitMeta)

def morrisScale():
    p = Problem(nInput=2, nObj=1, lb=0., ub=1., objFunc=lambda x:(x[:,0]+2*x[:,1])[:,None])
    x, meta = MorrisDesign().sampleWithMeta(p, 20, seed=2)
    y = p.evaluate(x).objs
    return {str(scale): Morris(**QUIET).analyze(p, x, y*scale, meta=meta).getMetric("S1_norm").values.tolist()
            for scale in [1.,1e-10]}
probe("morris_normalized_scale", morrisScale)

def moasmoNPop():
    return {str(n): dict(inner_npop=MOASMO(nPop=n, **QUIET).optimizer.get("nPop"))
            for n in [8,24,100]}
probe("moasmo_npop_unused", moasmoNPop)

def moasmoReuse():
    method = MOASMO(nInit=8, maxIters=1, maxFEs=30,
                     optimizer=NSGAII(nPop=8,maxIters=1,**QUIET), **QUIET)
    method.run(problem(2),seed=1)
    try:
        method.run(problem(3),seed=1)
    except Exception as e:
        return dict(second_run_error=str(e), type=type(e).__name__, fes_before_failure=method.FEs,
                    retained_model_count=method.surrogates.n_surrogates)
    return dict(second_run="succeeded")
probe("moasmo_reuse_output_count",moasmoReuse)

def algorithmMismatch():
    calls=[]
    p=Problem(nInput=2,nObj=2,lb=0.,ub=1.,
              objFunc=lambda x:(calls.append(len(x)) or np.column_stack([x[:,0],x[:,1]])))
    method=GA(nPop=8,maxIters=1,**QUIET)
    try:
        r=method.run(p,seed=1)
        return dict(status="accepted",best_shape=r.bestObjs.shape,evals=sum(calls))
    except Exception as e:
        return dict(error=type(e).__name__,message=str(e),evals=sum(calls))
probe("single_objective_algorithm_multi_objective_problem",algorithmMismatch)

def parameterSetter():
    method=GA(nPop=8,maxIters=1,**QUIET)
    method.set("maxIters",0)
    r=method.run(problem(),seed=1)
    return dict(get_value=method.get("maxIters"), actual_attribute=method.maxIter,
                actual_iterations=r.iters)
probe("budget_setter_split_state",parameterSetter)

def calibrationBounds():
    seen=[]
    p=ModelProblem(nInput=1,lb=0.,ub=1.,simFunc=lambda x:(seen.append(x.copy()) or x[:,None,:]),
                   obs=np.array([[10.]]))
    r=ES(**QUIET).run(p,np.array([[.2],[.8]]))
    return dict(posterior=r.posteriorDecs.tolist(),callback_inputs=[x.tolist() for x in seen],
                lower=p.lb.tolist(),upper=p.ub.tolist())
probe("calibration_bounds_not_applied",calibrationBounds)

def resultIdentity():
    method=GA(nPop=8,maxIters=0,**QUIET)
    r=method.run(problem(),seed=1)
    return dict(algorithm_run_id=method.runId,result_summary=r.summary(),extra_keys=list(r.extra))
probe("optimization_result_metadata_missing",resultIdentity)

def marsRefitDimension():
    model=MARS().fit(np.arange(20.)[:,None],np.arange(20.)[:,None])
    x=np.random.default_rng(1).random((30,2))
    try:
        model.fit(x,x.sum(axis=1)[:,None])
    except Exception as e:
        return dict(type=type(e).__name__,message=str(e),
                    fresh_model_succeeds=bool(np.all(np.isfinite(MARS().fit(x,x.sum(axis=1)[:,None]).predict(x)))))
    return dict(status="succeeded")
probe("mars_refit_changed_input_dimension",marsRefitDimension)

def multiPartialFit():
    x=np.arange(10.)[:,None]
    first,second=LinearRegression(),LinearRegression()
    model=MultiSurrogate(2,[first,second])
    y=np.hstack([x,2*x]);model.fit(x,y)
    old=model.predict(x).copy()
    def fail(*args):
        raise ValueError("controlled second-model failure")
    with patch.object(second,"fit",fail):
        try:model.fit(x,3*y)
        except ValueError:pass
    return dict(old=old[:3].tolist(),after_failure=model.predict(x)[:3].tolist())
probe("multi_surrogate_partial_refit",multiPartialFit)

def invalidCalibrationStillEvaluates():
    seen=[]
    p=ModelProblem(nInput=1,lb=0.,ub=1.,simFunc=lambda x:(seen.append(len(x)) or x[:,None,:]),obs=np.array([[.5]]))
    try: ES(**QUIET).run(p,np.array([[.3]]))
    except Exception as e:return dict(message=str(e),evaluated_rows=sum(seen))
probe("calibration_invalid_ensemble_late_validation",invalidCalibrationStillEvaluates)

def ensembleDecompositions():
    from UQPyL.calibration.methods._ensemble import ensembleGain
    counts={"eigh":0,"solve":0}
    originalEigh,originalSolve=np.linalg.eigh,np.linalg.solve
    def eigh(x): counts["eigh"]+=1;return originalEigh(x)
    def solve(x,y): counts["solve"]+=1;return originalSolve(x,y)
    with patch.object(np.linalg,"eigh",eigh),patch.object(np.linalg,"solve",solve):
        _,info=ensembleGain(np.ones((2,30)),np.eye(30),np.eye(30))
    return dict(calls=counts,info=info)
probe("ensemble_full_rank_double_decomposition",ensembleDecompositions)


def multiNaturalFailure():
    from UQPyL.surrogate.scaler import StandardScaler
    x=np.arange(10.)[:,None]
    first=LinearRegression()
    second=LinearRegression(scalers=(None,StandardScaler()))
    model=MultiSurrogate(2,[first,second])
    model.fit(x,np.hstack([x,2*x]))
    y=np.hstack([3*x,np.full_like(x,np.nan)])
    try:model.fit(x,y)
    except Exception as e:
        try:model.predict(x)
        except Exception as predError:
            return dict(fit_error=type(e).__name__,prediction_error=type(predError).__name__,
                        conclusion="prediction rejected; artificial override probe is not a built-in defect")
        return dict(conclusion="prediction unexpectedly succeeded")
    return dict(conclusion="probe inconclusive")
probe("countercheck_multi_surrogate_natural_failure",multiNaturalFailure)

def failedInferenceStorage():
    from tempfile import TemporaryDirectory
    from UQPyL.inference.runtime import InfReader
    with TemporaryDirectory() as directory:
        count=0
        def objective(x):
            nonlocal count
            count+=1
            if count==7: raise RuntimeError("controlled simulator failure")
            return np.sum(x*x,axis=1)[:,None]
        p=Problem(nInput=2,nObj=1,lb=0.,ub=1.,objFunc=objective)
        p.workDir=directory
        method=MH(nChains=1,warmUp=0,maxIters=10,verboseFlag=False,logFlag=False,saveFlag=True,saveFreq=2)
        try:method.run(p,seed=1)
        except RuntimeError:pass
        file=next(Path(directory).rglob("*.sqlite3"))
        with InfReader(file) as reader:
            snapshots=reader.list_snapshots()
            try:reader.load_result();message=None
            except Exception as e:message=str(e)
            return dict(status=reader.get_run()["status"],snapshot_iterations=[x["iter"] for x in snapshots],
                        load_result_error=message,observations_in_last_snapshot=len(reader.load_last_snapshot_members()))
probe("failed_inference_partial_result_availability",failedInferenceStorage)

def hvWork():
    import importlib
    module=importlib.import_module("UQPyL.optimization.runtime.result")
    original=module.HV
    calls=[]
    def timed(*args,**kwargs):
        start=perf_counter()
        result=original(*args,**kwargs)
        calls.append(dict(points=len(args[0]),n_samples=kwargs.get("nSamples",1_000_000),
                          seconds=perf_counter()-start))
        return result
    # Four tradeoff objectives guarantee a nontrivial nondominated set.
    p=Problem(nInput=2,nObj=4,lb=0.,ub=1.,objFunc=lambda x:np.column_stack([x[:,0],1-x[:,0],x[:,1],1-x[:,1]]))
    with patch.object(module,"HV",timed):
        start=perf_counter()
        result=NSGAII(nPop=12,maxIters=2,**QUIET).run(p,seed=1)
        elapsed=perf_counter()-start
    return dict(fes=result.FEs,total_seconds=elapsed,hv_calls=calls,
                hv_time_fraction=sum(c["seconds"] for c in calls)/elapsed)
probe("automatic_hv_runtime_cost",hvWork)

def negativePlot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from UQPyL.viz.surrogate import plot_surrogate
    with patch.object(plt,"show"):
        fig,ax=plot_surrogate("negative",np.array([-10.,-5.]),np.array([-10.,-5.]))
    limits=ax.get_xlim()
    plt.close(fig)
    return dict(data_min=-10.,data_max=-5.,axis_limits=list(limits),
                both_extreme_points_outside=(limits[0]>-10 and limits[1]<-5))
probe("negative_surrogate_plot_clipping",negativePlot)

def crossValidationAPI():
    import inspect
    from UQPyL.surrogate.auto_tuner import AutoTuner
    from UQPyL.inference import AMH
    from UQPyL.optimization.expensive import EGO
    return dict(opt_tune_signature=str(inspect.signature(AutoTuner.optTune)),
                grid_tune_signature=str(inspect.signature(AutoTuner.gridTune)),
                mh_signature=str(inspect.signature(MH)),amh_signature=str(inspect.signature(AMH)),
                ego_signature=str(inspect.signature(EGO)))
probe("public_config_capabilities",crossValidationAPI)

def calibrationCovarianceRepeats():
    nObs=20
    obs=np.linspace(.2,.6,nObs)[:,None]
    p=ModelProblem(nInput=1,lb=0.,ub=1.,simFunc=lambda x:np.repeat(x[:,None,:],nObs,axis=1),obs=obs)
    original=np.linalg.eigh; shapes=[]
    def eigh(matrix):
        shapes.append(matrix.shape)
        return original(matrix)
    with patch.object(np.linalg,"eigh",eigh):
        IES(maxIters=3,**QUIET).run(p,np.linspace(.1,.9,6)[:,None],r=np.eye(nObs)*.1)
    return dict(iterations=3,eigh_shapes=shapes,
                one_dense_matrix_bytes_at_10000_obs=10000*10000*8)
probe("ies_repeated_constant_covariance_validation",calibrationCovarianceRepeats)


def separateTuner():
    from UQPyL.surrogate.gp import GPR
    from UQPyL.surrogate.gp.kernel import RBF
    from UQPyL.surrogate.auto_tuner import AutoTuner
    x=np.linspace(0,1,24)[:,None]
    y=np.sin(6*x)
    result={}
    for mode in ["joint","separate"]:
        model=GPR(kernel=RBF(),C_attr=None,nRestartTimes=0)
        original=model._objfunc
        count=0
        def objective(*args,**kwargs):
            nonlocal count
            count+=1
            return original(*args,**kwargs)
        model._objfunc=objective
        best,score=AutoTuner(model).gridTune(x,y,paraGrid={"l":np.log([.1,.8])},ratio=25,seed=1,tuneMode=mode)
        result[mode]=dict(returned_parameter=np.asarray(best).tolist(),score=score,likelihood_calls=count)
    return result
probe("tuning_mode_compute_and_parameter_semantics",separateTuner)

def constrainedExpensive():
    from UQPyL.optimization.expensive import EGO
    seen=[]
    def objective(x):
        seen.append(x.copy())
        return x.copy()
    p=Problem(nInput=1,nObj=1,nCon=1,lb=0.,ub=1.,objFunc=objective,
              conFunc=lambda x:.9-x)
    method=EGO(nInit=4,maxIters=1,**QUIET)
    method.optimizer=GA(nPop=12,maxIters=3,**QUIET)
    method.run(p,seed=1,initialPop=np.array([[.1],[.3],[.6],[.9]]))
    return dict(new_evaluation=seen[-1].tolist(),
                new_violation=(.9-seen[-1]).tolist(),surrogate_targets="objectives only")
probe("ego_constraint_blind_acquisition",constrainedExpensive)

def savedInferenceSigns():
    from tempfile import TemporaryDirectory
    from UQPyL.inference.runtime import InfReader
    p=problem(optType="max")
    with TemporaryDirectory() as directory:
        p.workDir=directory
        model=MH(nChains=1,warmUp=0,maxIters=4,verboseFlag=False,logFlag=False,saveFlag=True,saveFreq=1)
        live=model.run(p,seed=1)
        with InfReader(next(Path(directory).rglob("*.sqlite3"))) as reader:
            loaded=reader.load_result()
            row=reader.load_last_snapshot_members()[0]
        return dict(live_last_obj=float(live.objs[0,-1,0]),loaded_last_obj=float(loaded.objs[0,-1,0]),
                    snapshot_last_obj=float(row["objs"][0]),
                    actual_last_obj=float(p.evaluate(live.decs[0,-1:]).objs[0,0]))
probe("inference_objective_sign_persists_to_sqlite",savedInferenceSigns)


def repeatedNumericChecks():
    observations=[]
    for seed in [1,3,7]:
        p=Problem(nInput=2,nObj=1,lb=[0.,0.],ub=[100.,1.],objFunc=lambda x:x.sum(axis=1)[:,None])
        values={}
        for output in ["real","unit"]:
            x,meta=SaltelliDesign(secondOrder=False).sampleWithMeta(p,256,seed=seed,output=output)
            values[output]=Sobol(**QUIET).analyze(p,x,meta=meta).getMetric("S1").values.tolist()
        mp=problem()
        mx,meta=MorrisDesign().sampleWithMeta(mp,20,seed=seed)
        y=mx[:,0:1]+2*mx[:,1:2]
        values["morris_small"]=Morris(**QUIET).analyze(mp,mx,y*1e-10,meta=meta).getMetric("S1_norm").values.tolist()
        values["seed"]=seed
        observations.append(values)
    return observations
probe("repeat_numeric_findings_across_seeds",repeatedNumericChecks)

path=Path(__file__).with_suffix(".json")
path.write_text(json.dumps(results,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+"\n")
print(json.dumps(results,ensure_ascii=False,indent=2,default=str))
