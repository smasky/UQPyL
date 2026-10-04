"""Bound layout, stable continuous conversions and single-sample snapshots."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from UQPyL.problem import Problem, ModelProblem, singleFunc
from UQPyL.doe import LHS


def makeProblem(lower,upper,model=False):
    cls=ModelProblem if model else Problem
    functions={'simFunc':lambda x:x.copy()} if model else {'objFunc':lambda x:x.sum(axis=1)[:,None]}
    return cls(nInput=np.asarray(lower).size,nObj=1,lb=lower,ub=upper,**functions)


@pytest.mark.parametrize('model',[False,True])
@pytest.mark.parametrize('shape',[(2,),(1,2),(2,1)])
@pytest.mark.parametrize('rows',[1,2,3])
def testBoundsHaveCanonicalShapeAndKeepSampleRows(model,shape,rows):
    lower=np.array([0.,10.]).reshape(shape)
    upper=np.array([1.,20.]).reshape(shape)
    p=makeProblem(lower,upper,model)
    assert p.lb.shape==p.ub.shape==(1,2)
    real=p.unit_to_space(np.full((rows,2),.5))
    np.testing.assert_array_equal(real,np.tile([.5,15.],(rows,1)))
    np.testing.assert_array_equal(p.space_to_unit(real),np.full((rows,2),.5))
    sample=LHS().sample(p,rows,seed=17)
    assert sample.shape==(rows,2)
    assert np.all(sample>=p.lb) and np.all(sample<=p.ub)
    lower.flat[0]=-99
    upper.flat[0]=99
    np.testing.assert_array_equal(p.lb,[[0,10]])
    np.testing.assert_array_equal(p.ub,[[1,20]])


@pytest.mark.parametrize('bound',[np.zeros((2,2)),np.zeros((1,1,4))])
def testAmbiguousBoundLayoutsAreRejected(bound):
    with pytest.raises(ValueError,match='bound'):
        Problem(nInput=4,nObj=1,lb=bound,ub=np.ones(4),objFunc=lambda x:x[:,:1])


@pytest.mark.parametrize('mode',['scalar','vector','view'])
def testSingleFuncSnapshotsReusedBuffers(mode):
    buffer=np.zeros(4)
    @singleFunc
    def objective(x):
        buffer[:]=[x[0],2*x[0],-x[0],4*x[0]]
        if mode=='scalar':
            return buffer[:1].reshape(())
        return buffer if mode=='vector' else buffer[::2]
    columns={'scalar':1,'vector':4,'view':2}[mode]
    p=Problem(nInput=1,nObj=columns,lb=0.,ub=5.,objFunc=objective)
    result=p.evaluate([[1.],[2.],[3.]])
    weights={'scalar':[1],'vector':[1,2,-1,4],'view':[1,-1]}[mode]
    np.testing.assert_array_equal(result.objs,np.arange(1,4)[:,None]*weights)
    buffer[:]=99
    np.testing.assert_array_equal(result.objs,np.arange(1,4)[:,None]*weights)


@pytest.mark.parametrize('lower,upper',[(-1e308,1e308),(-1.7e308,1e308),(-1e308,1.7e308),
                                       (-9e307,9e307),(1e308,1.7e308),(-1e-310,1e-310),(-3.,11.)])
def testContinuousConversionMatchesDecimal(lower,upper):
    p=makeProblem([lower],[upper])
    unit=np.array([0.,.125,.25,.5,.75,.875,1.])[:,None]
    original=unit.copy()
    with localcontext() as ctx:
        ctx.prec=1200
        lo,hi=Decimal.from_float(lower),Decimal.from_float(upper)
        expected=np.array([float(lo+(hi-lo)*Decimal.from_float(u)) for u in unit[:,0]])[:,None]
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        real=p.unit_to_space(unit)
        restored=p.space_to_unit(real)
    np.testing.assert_allclose(real,expected,rtol=3e-15,atol=np.nextafter(0.,1.)*2)
    np.testing.assert_allclose(restored,unit,rtol=0,atol=3e-14)
    np.testing.assert_array_equal(real[[0,-1]],[[lower],[upper]])
    np.testing.assert_array_equal(unit,original)


def testExtremeContinuousCoordinatesWithMixedAndFixedAxes():
    p=Problem(nInput=4,nObj=1,lb=[-1e308,-2,0,7],ub=[1e308,3,1,7],
              varType=[0,1,2,0],varSet={2:[.25,2.5,8.]},objFunc=lambda x:x[:,1:2])
    u=np.array([[0,0,0,0],[.5,.5,.5,.5],[1,1,1,1]])
    x=p.unit_to_space(u)
    np.testing.assert_array_equal(x,[[-1e308,-2,.25,7],[0,1,2.5,7],[1e308,3,8.,7]])
    np.testing.assert_array_equal(p.unit_to_space(p.space_to_unit(x)),x)
    np.testing.assert_array_equal(p.canonicalize_unit(u),p.space_to_unit(x))
