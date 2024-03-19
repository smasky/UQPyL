import numpy as np
from scipy.linalg import lstsq, solve
from .surrogate_ABC import Surrogate, Scale_T
from ..Utility.scalers import Scaler
from ..Utility.polynomial_features import PolynomialFeatures
from typing import Tuple, Literal, Optional
class LinearRegression(Surrogate):
    '''
    LinearRegression
    
    Support three type:
    'Origin'-------'Least Square Method'
    'Ridge'--------'Ridge'
    'Lasso'--------'Lasso'
    '''
    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                 poly_feature: PolynomialFeatures=None,
                 Type: Literal['Origin', 'Ridge', 'Lasso']='Origin',
                 fit_intercept: bool= True, alpha: float=0.1,
                 epoch: int=100, lr: float=1e-5, tl: float=1e-5):
        
        super().__init__(scalers, poly_feature)
        self.type=Type
        self.fit_intercept=fit_intercept
        self.alpha=alpha
        self.epoch=epoch
        self.lr=lr
        self.tl=tl
    ###########################Interface Function#############################
    def fit(self, trainX: np.ndarray, trainY: np.ndarray):
        
        trainX, trainY=self.__check_and_scale__(trainX, trainY)
        
        if self.type=='Origin':
            self._fit_Origin(trainX, trainY)
        elif self.type=='Ridge':
            self._fit_Ridge(trainX, trainY)
        elif self.type=='Lasso':
            self._fit_Lasso(trainX, trainY)
        else:
            raise ValueError('Using wrong model type!')
        
    def predict(self, predict_X: np.ndarray):
        
        predict_X=self.__X_transform__(predict_X)
        
        predict_Y=predict_X@self.coef_.T+self.intercept_
        
        return self.__Y_inverse_transform__(predict_Y)
    
    ###############################Private Function###############################
    def _fit_Origin(self, trainX: np.ndarray, trainY: np.ndarray):
        trainX_=trainX.copy()
        trainY_=trainY.copy()
        if self.fit_intercept:
            trainX_=np.hstack((trainX_, np.ones((trainX.shape[0],1))))
        
        self.coef_, _, self.rank_, self.singular_ = lstsq(trainX_, trainY_)
        
        if self.fit_intercept:
            self.intercept_=self.coef_[-1]
            self.coef_=(self.coef_[:-1].copy())
        else:
            self.coef_=self.coef_
        
    def _fit_Ridge(self, trainX: np.ndarray, trainY: np.ndarray):
        
        # w = inv(X^t X + alpha*Id) * X.T y
        trainX_=trainX.copy()
        trainY_=trainY.copy()
        
        _, n_features=trainX.shape
        
        if self.fit_intercept:
            offsetX=np.mean(trainX, axis=0)
            offsetY=np.mean(trainY, axis=0)
            trainX_-=offsetX
            trainY_-=offsetY
            
            
        trainX_.flat[::n_features+1]+=self.alpha
        A=np.dot(trainX_.T, trainX_)
        b=np.dot(trainX_.T, trainY_)
        
        self.coef_=solve(A, b)
        
        if self.fit_intercept:
            self.intercept_=offsetY-np.dot(offsetX.reshape(1,-1), self.coef_)
            return self.coef_, self.intercept_
        else:
            return self.coef_
    
    def _fit_Lasso(self, trainX: np.ndarray, trainY: np.ndarray):
        from .Lasso_ import celer
        from .Lasso_ import compute_norms_X_col, compute_Xw, dnorm_enet
        
        trainX_=np.asarray(trainX, order='F')
        trainY_=np.asarray(trainY, order='F')
        n_samples, n_features=trainX_.shape
        
        X_dense = trainX_
        X_data = np.empty([1], dtype=trainX_.dtype)
        X_indices = np.empty([1], dtype=np.int32)
        X_indptr = np.empty([1], dtype=np.int32)
        
        if self.fit_intercept:
            offsetX=np.mean(trainX_, axis=0)
            offsetY=np.mean(trainY_, axis=0)
            trainX_-=offsetX
            trainY_-=offsetY
            
            X_sparse_scaling=offsetX
        else:
            X_sparse_scaling = np.zeros(n_features, dtype=trainX_.dtype)
        
        norms_X_col=np.zeros(n_features, dtype=X_dense.dtype)
        compute_norms_X_col(
            False, norms_X_col, n_samples, X_dense, X_data,
            X_indices, X_indptr, X_sparse_scaling)
        
        w=np.zeros(n_features, dtype=X_dense.dtype)
        Xw=np.zeros(n_samples, dtype=X_dense.dtype)
        compute_Xw(False, 0, Xw, w, trainY_.ravel(), X_sparse_scaling.any(), X_dense,
                    X_data, X_indices, X_indptr, X_sparse_scaling)
        theta=Xw.copy()
        
        weights=np.ones(n_features, dtype=X_dense.dtype)
        positive=False
        if self.alpha:
            alpha=self.alpha
        else:
            alpha=1.0
        l1_ratio=1.0

        skip = np.zeros(trainX_.shape[1], dtype=np.int32)
        dnorm = dnorm_enet(False, theta, w, X_dense, X_data, 
                           X_indices, X_indptr,skip, X_sparse_scaling, 
                           weights, X_sparse_scaling.any(), positive,
                           alpha, l1_ratio)
        
        theta /= max(dnorm / (alpha * l1_ratio), n_samples)
        
        max_iter=100; max_epochs=500000; p0=10
        verbose=0; tol=0.0001; prune=True
        sol = celer(False, 0,X_dense, X_data, X_indices, 
                    X_indptr, X_sparse_scaling, trainY_.ravel(),
                    self.alpha, l1_ratio, w, Xw, 
                    theta, norms_X_col, weights,
                    max_iter=max_iter, max_epochs=max_epochs,
                    p0=p0, verbose=verbose, use_accel=1, tol=tol, prune=prune,
                    positive=positive)
        self.coef_=sol[0]

        #####Coordinate Descent
        # w = np.ones((n_features,1))*0.2
        # for _ in range(self.epoch):
        #     pre_w=w.copy()   
        #     for i in range(n_features):
        #         for n in range(self.epoch):
        #             Y_hat=np.dot(trainX_, w)
        #             g_i=np.dot(trainX_[:, i].T,(Y_hat-trainY_))/n_samples+ self.alpha*np.sign(w[i])
        #             w[i]=w[i]- g_i*self.lr          
        #             if np.abs(g_i)<self.tl:
        #                 break   
        #     diff_w= np.array(list(map(lambda x: abs(x)<self.tl, pre_w-w)))
        #     if diff_w.all():
        #         break
        if self.fit_intercept:
            self.intercept_=offsetY-np.dot(offsetX.reshape(1,-1), self.coef_)
            return self.coef_, self.intercept_
        else:
            return self.coef_
        
    def predict(self, predict_X: np.ndarray):
        
        predict_X=self.__X_transform__(predict_X)
        
        if(self.fit_intercept):
            predict_Y=predict_X@self.coef_+self.intercept_
        else:
            predict_Y=predict_X@self.coef_
        
        return self.__Y_inverse_transform__(predict_Y)
    
    ############################Attribute#########################
    @property
    def coef(self):
        return self.coef_
    
    @coef.setter
    def coef(self, value):
        self.coef_=value
    
    @property
    def intercept(self):
        return self.intercept_
    
    @intercept.setter
    def intercept(self, value):
        self.intercept_=value
    
    @property
    def alpha(self):
        return self.alpha_
    
    @alpha.setter
    def alpha(self, value):
        self.alpha_=value
        
        
            
        
        
        
        

