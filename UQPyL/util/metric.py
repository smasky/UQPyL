import numpy as np

def r_square(true_Y: np.ndarray, pre_Y: np.ndarray) -> np.ndarray:
    """
    R2-score
    """
    SSR = np.sum(np.square(true_Y-pre_Y))
    mean_Y = np.mean(true_Y, axis=0)
    SST = np.sum(np.square(true_Y-mean_Y))
    
    return 1-SSR/SST

def nse(true_Y: np.ndarray, pre_Y: np.ndarray) -> np.ndarray:
    """
    NSE
    """
    return 1-np.sum(np.square(true_Y-pre_Y))/np.sum(np.square(true_Y-np.mean(true_Y, axis=0)))

def mse(true_Y: np.ndarray, pre_Y: np.ndarray) -> np.ndarray:
    """
    Mean square error
    """
    return np.mean(np.square(true_Y-pre_Y), axis=0)

def rank_score(true_Y: np.ndarray, pre_Y: np.ndarray) -> np.ndarray:
    """
    Rank score-Kendall rank
    """
    
    ty = true_Y.ravel()
    py = pre_Y.ravel()
    n_samples = true_Y.shape[0]
    
    nc = 0; nd = 0
    for n in range(n_samples-1):
        for m in range(n+1, n_samples):
            sign = (ty[n]-ty[m])*(py[n]-py[m])
            if(sign>0 or (ty[n]==ty[m] and py[n]==py[m])):
                nc+=1
            else :
                nd+=1
                
    return (nc-nd)/(n_samples*(n_samples-1))*2


def sort_score(true_Y: np.ndarray, pre_Y: np.ndarray) -> np.ndarray:
    """
    Sort_score
    """
    
    t_idx = np.argsort(true_Y.ravel())
    p_idx = np.argsort(pre_Y.ravel())
    
    return np.sum(np.abs(t_idx-p_idx))
    
    