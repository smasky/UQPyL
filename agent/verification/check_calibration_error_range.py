"""Independent high-precision sweep of physical error metrics."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import warnings

import numpy as np

from UQPyL.calibration.util import mse, mae, rmse

records=[]
for seed in [11,23,47]:
    rng=np.random.default_rng(seed)
    baseObs=rng.normal(size=17)
    baseSim=rng.normal(size=(3,17))
    for exponent in [-300,-160,-150,-80,0,80,150,154,200,300]:
        obs,sim=baseObs*10.**exponent,baseSim*10.**exponent
        for name,function in [('MSE',mse),('MAE',mae),('RMSE',rmse)]:
            expected=[]
            with localcontext() as context:
                context.prec=120
                for row in sim:
                    residual=[Decimal.from_float(float(a))-Decimal.from_float(float(b)) for a,b in zip(row,obs)]
                    moment=sum(abs(x) if name=='MAE' else x*x for x in residual)/len(row)
                    expected.append(float(moment.sqrt() if name=='RMSE' else moment))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                actual=function(obs,sim)
            np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=0)
            outOfRange=any(value==0 or not np.isfinite(value) for value in expected)
            assert bool(caught)==outOfRange
            for index,(value,reference) in enumerate(zip(actual,expected)):
                relativeError=float(abs(value/reference-1)) if reference and np.isfinite(reference) else None
                records.append(dict(seed=seed,scale_exponent=exponent,metric=name,row=index,
                                    actual=str(float(value)),expected=str(reference),relative_error=relativeError,
                                    warnings=[str(w.message) for w in caught]))
path=Path(__file__).with_name('1002-calibration-error-range.json')
path.write_text(json.dumps(records,indent=2)+'\n')
print('records',len(records))
print('max relative error',max(r['relative_error'] for r in records if r['relative_error'] is not None))
