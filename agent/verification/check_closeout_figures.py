"""Render representative public plots and verify their stored numerical data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from UQPyL.analysis import Morris
from UQPyL.doe import MorrisDesign
from UQPyL.inference import MH
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem
from UQPyL.viz import plot_sa, plot_op_curve_stat, plot_op_pareto, plot_infer_trace, plot_infer_stat_combined, plot_surrogate
import warnings

output=Path('agent/verification/1004-figures');output.mkdir(exist_ok=True)
plt.show=lambda:None
records=[]

def save(name,fig,**checks):
    fig.savefig(output/(name+'.png'),dpi=80,bbox_inches='tight')
    plt.close(fig)
    records.append(dict(plot=name,**checks))

p=Problem(nInput=3,nObj=1,lb=0.,ub=1.,objFunc=lambda x:x[:,:1]-2*x[:,1:2]+4*x[:,2:3])
x,meta=MorrisDesign().sampleWithMeta(p,20,seed=17)
result=Morris(verboseFlag=False).analyze(p,x,meta=meta)
fig,ax=plot_sa({'Morris':result},metric='mu',title='Signed Morris effects',yLabel='Elementary effect')
np.testing.assert_allclose([b.get_height() for b in ax.patches],[1,-2,4],atol=1e-14)
save('sensitivity',fig,values=[b.get_height() for b in ax.patches])

a=GA(nPop=4,maxIters=0,saveFlag=False,verboseFlag=False).run(p,seed=17)
b=GA(nPop=4,maxIters=0,saveFlag=False,verboseFlag=False).run(p,seed=17)
a.history.iterToFEs=[[0,10],[1,20],[2,30]];a.history.bestObjHistory=[-2,-4,-6]
b.history.iterToFEs=[[0,20],[1,30],[2,40]];b.history.bestObjHistory=[-8,-10,-12]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    fig,ax=plot_op_curve_stat({'Two runs':[a,b]},xCoord='fe',xLabel='Model evaluations')
np.testing.assert_array_equal(ax.lines[0].get_ydata(),[-6,-8])
save('optimization',fig,x=ax.lines[0].get_xdata().tolist(),warnings=[str(w.message) for w in caught])

a.bestObjs=np.array([[0,1],[.3,.7],[1,0.]])
fig,ax=plot_op_pareto(a,optima=np.array([[0,1],[.5,.5],[1,0]]),title='Pareto points and reference')
save('pareto',fig,reference_x=ax.lines[0].get_xdata().tolist())

r=MH(nChains=3,warmUp=10,maxIters=300,saveFlag=False,verboseFlag=False).run(p,seed=17)
fig,axes=plot_infer_trace(r,idx=[2,0],burnIn=10)
assert axes[0].get_title()=='Decision Variable 3'
save('trace',fig,titles=[ax.get_title() for ax in axes])
fig,axes=plot_infer_stat_combined(r,burnIn=10,showCI=True)
save('distribution',fig,dimensions=3)

truth=np.array([-3.,-1,1,3]);prediction=truth+np.array([.1,-.1,.2,-.2])
fig,ax=plot_surrogate('Independent holdout',prediction,truth)
np.testing.assert_array_equal(ax.collections[0].get_offsets(),np.column_stack([truth,prediction]))
save('surrogate',fig,points=4)

canvas=Image.new('RGB',(1440,960),'white')
for index,row in enumerate(records):
    with Image.open(output/(row['plot']+'.png')) as picture:
        picture.thumbnail((480,480))
        canvas.paste(picture,((index%3)*480,(index//3)*480))
canvas.save(output/'preview.png')
(output/'checks.json').write_text(json.dumps(records,indent=2))
print('Rendered and numerically verified',len(records),'plots')
