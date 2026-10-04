"""Focused re-review of state ownership and recently touched public entry points."""
import json
from pathlib import Path
import numpy as np
from UQPyL.surrogate import MultiSurrogate
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF
from UQPyL.surrogate.scaler import StandardScaler
from UQPyL.surrogate.auto_tuner import AutoTuner
from UQPyL.surrogate.mars import MARS

results={}
def fixed(**kw):
    return GPR(kernel=RBF(length_scale=.3,length_attr=None),C_attr=None,**kw)
def probe(name,func):
    try: results[name]=func()
    except Exception as error: results[name]={'unexpected_error':type(error).__name__,'message':str(error)}

def tuningFailure():
    x=np.linspace(0,1,6)[:,None]; y=np.sin(3*x)
    m=fixed(scalers=(StandardScaler(),None)).fit(x,y)
    before=m.predict(x)
    try:
        AutoTuner(m).gridTune(np.linspace(10,20,8)[:,None],np.arange(8.)[:,None],paraGrid={},ratio=25,seed=1)
    except ValueError as error:
        message=str(error)
    after=m.predict(x)
    return {'error':message,'prediction_delta':float(np.max(abs(before-after))),'state_keys':list(m.fitState)}
probe('autotuner_empty_grid_corrupts_previous_fit',tuningFailure)

def duplicateModels():
    x=np.linspace(0,1,8)[:,None]; y=np.hstack([np.sin(3*x),np.cos(3*x)])
    shared=fixed(); model=MultiSurrogate(2,[shared,shared]); model.fit(x,y)
    pred=model.predict(x)
    return {'outputs_identical':bool(np.array_equal(pred[:,0],pred[:,1])), 'first_output_max_error':float(np.max(abs(pred[:,0]-y[:,0])))}
probe('duplicate_multisurrogate_instance',duplicateModels)

def marsFailure():
    x=np.linspace(0,1,30)[:,None]; y=3*x+2
    model=MARS().fit(x,y)
    before=model.predict_deriv(x).copy()
    try: model.fit(x,y[:-1])
    except ValueError as error: message=str(error)
    try: model.predict(x); prediction='succeeded'
    except RuntimeError: prediction='rejected'
    derivative=model.predict_deriv(x)
    return {'fit_error':message,'predict_after_failure':prediction,'derivative_still_returns_old_values':bool(np.array_equal(before,derivative))}
probe('mars_derivative_after_failed_fit',marsFailure)

def marsScaling():
    x=np.linspace(10,20,30)[:,None]; y=3*x+2
    model=MARS(scalers=(StandardScaler(),StandardScaler())).fit(x,y)
    query=np.array([[13.],[17.]])
    analytical=model.predict_deriv(query).ravel()
    finite=((model.predict(query+1e-4)-model.predict(query-1e-4))/2e-4).ravel()
    return {'derivative_api':analytical.tolist(),'finite_difference_of_predict':finite.tolist()}
probe('mars_derivative_scaling',marsScaling)
def marsSampleScores():
    x=np.linspace(0,1,30)[:,None]; y=3*x+2
    model=MARS().fit(x,y)
    try:
        model.score_samples(x,y)
    except TypeError as error:
        return {'error':str(error)}
probe('mars_score_samples_call',marsSampleScores)

def unknownGridKey():
    from UQPyL.surrogate.regression.linear_regression import LinearRegression
    x=np.linspace(0,1,20)[:,None]; y=x*x+1
    model=LinearRegression(lossType='Ridge')
    parameters,score=AutoTuner(model).gridTune(x,y,{'misspelled_parameter':[.1,.2]},ratio=25,seed=1,tuneMode='joint')
    return {'returned_parameter':parameters,'score':float(score)}
probe('unrecognized_grid_key',unknownGridKey)

Path('agent/verification/0919-followup-review.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
