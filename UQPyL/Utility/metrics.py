import numpy as np
import math
def r2_score(true_Y: np.ndarray, pre_Y: np.ndarray):
    """
    R2-score
    """
    SSR=np.sum(np.square(true_Y-pre_Y))
    mean_Y=np.mean(true_Y, axis=0)
    SST=np.sum(np.square(true_Y-mean_Y))
    
    return 1-SSR/SST

def mse(true_Y: np.ndarray, pre_Y: np.ndarray):
    """
    Mean square error
    """
    return np.mean(np.square(true_Y-pre_Y), axis=0)

def rank_score(true_Y: np.ndarray, pre_Y: np.ndarray):
    """
    Rank score-Kendall rank
    """
    
    ty=true_Y.ravel()
    py=pre_Y.ravel()
    count=0
    n_samples=true_Y.shape[0]
    
    nc=0;nd=0;nt1=0;nt2=0
    for n in range(n_samples-1):
        for m in range(n+1, n_samples):
            sign=(ty[n]-ty[m])*(py[n]-py[m])
            if(sign>0):
                nc+=1
            else :
                nd+=1
                
    return (nc-nd)/(n_samples*(n_samples-1))*2


def sort_score(true_Y: np.ndarray, pre_Y: np.ndarray):
    """
    Sort_score
    """
    
    t_idx=np.argsort(true_Y.ravel())
    p_idx=np.argsort(pre_Y.ravel())
    
    return np.sum(np.abs(t_idx-p_idx))
    
    