import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Union

from .core._forward import ForwardPasser
from .core._pruning import PruningPasser
from .core._util import ascii_table, apply_weights_2d, gcv
from .core._types import BOOL
from ..base import SurrogateABC
from ..scaler import Scaler
from ..poly import PolyFeature

class MARS(SurrogateABC):

    """
    Multivariate Adaptive Regression Splines (MARS) surrogate model.

    This implementation is adapted from the py-earth style workflow. MARS is a
    flexible nonparametric regression method that automatically discovers
    nonlinear effects and low-order interactions through forward basis
    construction followed by pruning.

    The training workflow has two stages:
    - forward pass: grow basis functions greedily
    - pruning pass: remove redundant terms using generalized cross-validation

    Examples:
        >>> model = MARS(max_terms=100, max_degree=2)
        >>> model.fit(xTrain, yTrain)
        >>> yPred = model.predict(xPred)

    References:
        [1] Friedman, Jerome. Multivariate Adaptive Regression Splines.
            Annals of Statistics. Volume 19, Number 1 (1991), 1-67.
        [2] Fast MARS, Jerome H.Friedman, Technical Report No.110, May 1993.
        [3] Estimating Functions of Mixed Ordinal and Categorical Variables
            Using Adaptive Splines, Jerome H.Friedman, Technical Report
            No.108, June 1991.
        [4] http://www.milbo.org/doc/earth-notes.pdf
    """

    forward_pass_arg_names = [
        'max_terms', 'max_degree', 'allow_missing', 'penalty',
        'endspan_alpha', 'endspan',
        'minspan_alpha', 'minspan',
        'thresh', 'zero_tol', 'min_search_points',
        'check_every', 'allow_linear',
        'use_fast', 'fast_K', 'fast_h',
        'feature_importance_type',
        'verbose'
    ]
    
    pruning_pass_arg_names = set([
        'penalty',
        'feature_importance_type',
        'verbose'
    ])
    defaultTuneParameters = ("max_terms", "max_degree", "penalty")
    advancedTuneParameters = ("endspan", "minspan", "thresh")

    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 polyFeature: PolyFeature = None, 
                 max_terms: Union[int] = 400, 
                 max_terms_attr: Union[dict, None] = {'ub': 400, 'lb': 10, 'type': 'int', 'log': False},
                 max_degree: int = 1, 
                 max_degree_attr: Union[dict, None] = {'ub': 10, 'lb': 1, 'type': 'int', 'log': False},
                 penalty: float = 3.0,
                 penalty_attr: Union[dict, None] = {'ub': 10, 'lb': 1, 'type': 'float', 'log': False},
                 endspan_alpha: float = 0.05,
                 endspan: int = -1,
                 minspan_alpha: float = 0.05,
                 minspan: int = -1,
                 thresh: float = 0.001,
                 zero_tol: float = 1e-12,
                 min_search_points: int = 100,
                 check_every: int = -1,
                 allow_linear: bool = True,
                 use_fast: bool = False,
                 fast_K: int = 5,
                 fast_h: int = 1,
                 smooth: bool = False,
                 enable_pruning: bool = True,
                 feature_importance_type: str = 'gcv'):
        """
        Initialize the MARS surrogate model.

        Args:
            scalers: Optional input/output scalers.
            polyFeature: Optional preprocessing transform applied before fitting.
            max_terms: Maximum number of basis terms generated in the forward pass.
            max_terms_attr: Tuning metadata for `max_terms`.
            max_degree: Maximum interaction degree of generated basis terms.
            max_degree_attr: Tuning metadata for `max_degree`.
            penalty: Complexity penalty used in GCV-based pruning.
            penalty_attr: Tuning metadata for `penalty`.
            endspan_alpha: Probabilistic control for automatic `endspan`.
            endspan: Number of edge samples excluded from knot selection. Use `-1`
                to infer it from `endspan_alpha`.
            minspan_alpha: Probabilistic control for automatic `minspan`.
            minspan: Minimum spacing between knots. Use `-1` to infer it from
                `minspan_alpha`.
            thresh: Forward-pass stopping threshold.
            zero_tol: Numerical zero tolerance used in the forward pass.
            min_search_points: Sample-count threshold used when deriving
                `check_every`.
            check_every: Candidate-knot subsampling interval. Use `-1` to infer it.
            allow_linear: Whether the forward pass may keep knotless linear terms.
            use_fast: Whether to use the approximate fast-MARS search strategy.
            fast_K: Parent-term shortlist size used by fast-MARS.
            fast_h: Revisit interval for full variable search in fast-MARS.
            smooth: Whether to smooth the final basis for continuous first derivatives.
            enable_pruning: Whether to run the pruning pass after forward fitting.
            feature_importance_type: Feature-importance criterion name.
        """
        
        super().__init__(scalers, polyFeature)
        
        
        allow_missing = False
        verbose = 0
        
        self.setting.set("max_terms", max_terms, max_terms_attr)
        self.setting.set("max_degree", max_degree, max_degree_attr)
        self.setting.set("penalty", penalty, penalty_attr)
        
        self.setting.set("endspan_alpha", endspan_alpha)
        self.setting.set("endspan", endspan)
        self.setting.set("minspan_alpha", minspan_alpha)
        self.setting.set("minspan", minspan)
        self.setting.set("thresh", thresh)
        self.setting.set("zero_tol", zero_tol)
        self.setting.set("min_search_points", min_search_points)
        self.setting.set("check_every", check_every)
        self.setting.set("allow_linear", allow_linear)
        self.setting.set("use_fast", use_fast)
        self.setting.set("fast_K", fast_K)
        self.setting.set("fast_h", fast_h)
        self.setting.set("smooth", smooth)
        self.setting.set("enable_pruning", enable_pruning)
        self.setting.set("feature_importance_type", feature_importance_type)
        self.setting.set("verbose", verbose)
        self.setting.set("allow_missing", allow_missing)
        
#-------------------------Public Function---------------------------#
    def fitModel(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        Fit the MARS model on prepared training data.

        The workflow is:
        1. scrub and validate inputs
        2. run forward pass
        3. optionally prune
        4. optionally smooth the basis
        5. solve the final linear system
        """
        self.resetFitState()
        self.storeTrainingData(xTrain, yTrain)

        #indicate label for each dimension
        self.xlabels_ = self._scrape_labels(xTrain)
        xTrain, yTrain, sample_weight, output_weight, missing = self._scrub(
            xTrain, yTrain, None, None, None)
        #forward
        self.forward_pass(xTrain, yTrain,
                          sample_weight, output_weight, missing,
                          self.xlabels_, [], skip_scrub=True)
        #pruning
        if self.setting.get("enable_pruning") is True:
            self.pruning_pass(xTrain, yTrain,
                              sample_weight, output_weight, missing,
                              skip_scrub=True)
        if self.setting.get("smooth"):
            self.basis_ = self.basis_.smooth(xTrain)
        self.linear_fit(xTrain, yTrain, sample_weight, output_weight, missing,
                        skip_scrub=True)
        self.fitState["basis"] = self.basis_
        self.fitState["coef"] = self.coef_
        self.fitState["mse"] = self.mse_
        self.fitState["gcv"] = self.gcv_
        self.fitState["rsq"] = self.rsq_
        self.fitState["grsq"] = self.grsq_
        return self

    def fitHyper(self, xTrain: np.ndarray, yTrain: np.ndarray):
        """
        Fit MARS under the current parameter setting.

        MARS does not use a separate internal hyper-optimization stage here, so
        `fitHyper` delegates directly to `fitModel`.
        """
        return self.fitModel(xTrain, yTrain)
    
    def predict(self, xPredict: np.ndarray, returnStd: bool = False,
                returnVar: bool = False):
        """
        Predict outputs for new samples.

        Args:
            xPredict: Input samples to evaluate.
            returnStd: Unsupported for MARS; kept for base-class consistency.
            returnVar: Unsupported for MARS; kept for base-class consistency.

        Returns:
            Predicted outputs in the original target scale.
        """
        self._normalize_predict_flags(returnStd, returnVar)
        self.requireFitted("basis", "coef")
        
        xPredict = self.__X_transform__(xPredict)
        
        X, missing = self._scrub_x(xPredict, None)
        B = self.transform(X, missing)
        y = np.dot(B, self.coef_.T)
        
        return self.__Y_inverse_transform__(y)

    def getDefaultTuneParameters(self, advanced: bool = False):
        """
        Return the default tunable parameter names.

        Args:
            advanced: Whether to include secondary tuning parameters.
        """
        params = list(self.defaultTuneParameters)
        if advanced:
            params.extend(self.advancedTuneParameters)
        return params

#------------------------------Private Function-------------------------#
    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return False
        keys = set(self.__dict__.keys()) | set(other.__dict__.keys())
        for k in keys:
            try:
                v_self = self.__dict__[k]
                v_other = other.__dict__[k]
            except KeyError:
                return False
            try:
                if v_self != v_other:
                    return False
            except ValueError:  # Case of numpy arrays
                if np.any(v_self != v_other):
                    return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def _pull_forward_args(self, setting):
        '''
        Pull named arguments relevant to the forward pass.
        '''
        result = {}
        for name in self.forward_pass_arg_names:
            if name in setting.parVal.keys():
                result[name] = self._normalize_cython_arg(setting.parVal[name])
            elif name in setting.parCon.keys():
                result[name] = self._normalize_cython_arg(setting.parCon[name])
        return result

    def _pull_pruning_args(self, setting):
        '''
        Pull named arguments relevant to the pruning pass.
        '''
        result = {}
        for name in self.pruning_pass_arg_names:
            if name in setting.parVal.keys():
                result[name] = self._normalize_cython_arg(setting.parVal[name])
            elif name in setting.parCon.keys():
                result[name] = self._normalize_cython_arg(setting.parCon[name])
        return result

    def _normalize_cython_arg(self, value):
        if isinstance(value, np.ndarray) and value.size == 1:
            return value.item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _scrape_labels(self, X):
        '''
        Try to get labels from input data (for example, if X is a
        pandas DataFrame).  Return None if no labels can be extracted.
        '''
        try:
            labels = list(X.columns)
        except AttributeError:
            try:
                labels = list(X.design_info.column_names)
            except AttributeError:
                try:
                    labels = list(X.dtype.names)
                except TypeError:
                    try:
                        labels = ['x%d' % i for i in range(X.shape[1])]
                    except IndexError:
                        labels = ['x%d' % i for i in range(1)]
                # handle case where X is not np.array (e.g list)
                except AttributeError:
                    X = np.array(X)
                    labels = ['x%d' % i for i in range(X.shape[1])]
        return labels

    def _scrub_x(self, X, missing, **kwargs):
        '''
        Sanitize input predictors and extract column names if appropriate.
        '''
        # Check for sparseness
        if sparse.issparse(X):
            raise TypeError('A sparse matrix was passed, but dense data '
                            'is required. Use X.toarray() to convert to '
                            'dense.')
        X = np.asarray(X, dtype=np.float64, order='F')
        
        # Figure out missingness
        missing_is_nan = False
        if missing is None:
            # Infer missingness
            missing = np.isnan(X)
            missing_is_nan = True
            
        if X.ndim == 1:
            X = X[:, np.newaxis]

        # Ensure correct number of columns
        if hasattr(self, 'basis_') and self.basis_ is not None:
            if X.shape[1] != self.basis_.num_variables:
                raise ValueError('Wrong number of columns in X. Reshape your data.')
        
        # Zero-out any missing spots in X
        if np.any(missing):
            if not self.setting.get("allow_missing"):
                raise ValueError('Missing data requires allow_missing=True.')
            if missing_is_nan or np.any(np.isnan(X)):
                X = X.copy()
                X[missing] = 0.
        
        # Convert to internally used data type
        missing = np.asarray(missing, dtype=BOOL, order='F')
        # assert_all_finite(missing)
        if missing.ndim == 1:
            missing = missing[:, np.newaxis]
        
        return X, missing

    def _scrub(self, X, y, sample_weight, output_weight, missing, **kwargs):
        '''
        Sanitize input data.
        '''
        # Check for sparseness
        if sparse.issparse(y):
            raise TypeError('A sparse matrix was passed, but dense data '
                            'is required. Use y.toarray() to convert to '
                            'dense.')
        if sparse.issparse(sample_weight):
            raise TypeError('A sparse matrix was passed, but dense data '
                            'is required. Use sample_weight.toarray()'
                            'to convert to dense.')
        if sparse.issparse(output_weight):
            raise TypeError('A sparse matrix was passed, but dense data '
                            'is required. Use output_weight.toarray()'
                            'to convert to dense.')

        # Check whether X is the output of patsy.dmatrices
        if y is None and isinstance(X, tuple):
            y, X = X

        # Handle X separately
        X, missing = self._scrub_x(X, missing, **kwargs)

        # Convert y to internally used data type
        y = np.asarray(y, dtype=np.float64)
        # assert_all_finite(y)

        if len(y.shape) == 1:
            y = y[:, np.newaxis]

        # Deal with sample_weight
        if sample_weight is None:
            sample_weight = np.ones((y.shape[0], 1), dtype=y.dtype)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            # assert_all_finite(sample_weight)
            if len(sample_weight.shape) == 1:
                sample_weight = sample_weight[:, np.newaxis]
        # Deal with output_weight
        if output_weight is not None:
            output_weight = np.asarray(output_weight, dtype=np.float64)
            # assert_all_finite(output_weight)

        # Make sure dimensions match
        if y.shape[0] != X.shape[0]:
            raise ValueError('X and y do not have compatible dimensions.')
        if y.shape[0] != sample_weight.shape[0]:
            raise ValueError(
                'y and sample_weight do not have compatible dimensions.')
        if output_weight is not None and y.shape[1] != output_weight.shape[0]:
            raise ValueError(
                'y and output_weight do not have compatible dimensions.')
        if y.shape[1] > 1:
            if sample_weight.shape[1] == 1 and output_weight is not None:
                sample_weight = np.repeat(sample_weight, y.shape[1], axis=1)
        if output_weight is not None:
            sample_weight *= output_weight

        return X, y, sample_weight, None, missing
    
    def forward_pass(self, X, y=None,
                     sample_weight=None, output_weight=None,
                     missing=None,
                     xlabels=None, linvars=[], skip_scrub=False):
        """
        Run the forward basis-construction stage.

        This stage greedily grows candidate basis functions before pruning.
        """
        
        # Label and format data
        if xlabels is None:
            self.xlabels_ = self._scrape_labels(X)
        else:
            self.xlabels_ = xlabels
        if not skip_scrub:
            X, y, sample_weight, output_weight, missing = self._scrub(
                X, y, sample_weight, output_weight, missing)

        # Do the actual work
        args = self._pull_forward_args(self.setting)
        
        forward_passer = ForwardPasser(
            X, missing, y, sample_weight,
            xlabels=self.xlabels_, linvars=linvars, **args)
        forward_passer.run()
        self.forward_pass_record_ = forward_passer.trace()
        self.basis_ = forward_passer.get_basis()

    def pruning_pass(self, X, y=None, sample_weight=None, output_weight=None,
                     missing=None, skip_scrub=False):
        """
        Run the pruning stage on an existing forward-pass basis.

        The pruning pass removes redundant terms using the configured
        complexity penalty and feature-importance criteria.
        """

        # Format data
        if not skip_scrub:
            X, y, sample_weight, output_weight, missing = self._scrub(
                X, y, sample_weight, output_weight, missing)

        # Pull arguments from self
        args = self._pull_pruning_args(self.setting)

        # Do the actual work
        pruning_passer = PruningPasser(
            self.basis_, X, missing, y, sample_weight,
            **args)
        pruning_passer.run()

        imp = pruning_passer.feature_importance
        self._feature_importances_dict = imp
        if len(imp) == 1: # if only one criterion then return it only
            imp = imp[list(imp.keys())[0]]
        elif len(imp) == 0:
            imp = None
        self.feature_importances_ = imp
        self.pruning_pass_record_ = pruning_passer.trace()

    def forward_trace(self):
        """Return the stored forward-pass trace, or `None` if unavailable."""
        try:
            return self.forward_pass_record_
        except AttributeError:
            return None

    def pruning_trace(self):
        """Return the stored pruning-pass trace, or `None` if unavailable."""
        try:
            return self.pruning_pass_record_
        except AttributeError:
            return None

    def trace(self):
        """Return a combined trace object for forward and pruning stages."""
        return EarthTrace(self.forward_trace(), self.pruning_trace())

    def summary(self):
        """Return a human-readable summary of the fitted model."""
        result = ''
        if self.forward_trace() is None:
            result += 'Untrained Earth Model'
            return result
        elif self.pruning_trace() is None:
            result += 'Unpruned Earth Model\n'
        else:
            result += 'Earth Model\n'
        header = ['Basis Function', 'Pruned']
        if self.coef_.shape[0] > 1:
            header += ['Coefficient %d' %
                       i for i in range(self.coef_.shape[0])]
        else:
            header += ['Coefficient']
        data = []

        i = 0
        for bf in self.basis_:
            data.append([str(bf), 'Yes' if bf.is_pruned() else 'No'] + [
                          '%g' % self.coef_[c, i] if not bf.is_pruned() else
                          'None' for c in range(self.coef_.shape[0])])
            if not bf.is_pruned():
                i += 1
        result += ascii_table(header, data)
        result += '\n'
        result += 'MSE: %.4f, GCV: %.4f, RSQ: %.4f, GRSQ: %.4f' % (
            self.mse_, self.gcv_, self.rsq_, self.grsq_)
        return result

    def summary_feature_importances(self, sort_by=None):
        """
        Return a formatted feature-importance table.

        Args:
            sort_by: Optional criterion name used to sort features.
        """
       
        result = ''
        if self._feature_importances_dict:
            max_label_length = max(map(len, self.xlabels_)) + 5
            result += (max_label_length * ' ' +
                       '    '.join(self._feature_importances_dict.keys()) + '\n')
            labels = np.array(self.xlabels_)
            if sort_by:
                if sort_by not in self._feature_importances_dict.keys():
                    raise ValueError('Invalid feature importance type name '
                                     'to sort with : %s, available : %s' % (
                                         sort_by,
                                         self._feature_importances_dict.keys()))
                imp = self._feature_importances_dict[sort_by]
                indices = np.argsort(imp)[::-1]
            else:
                indices = np.arange(len(labels))
            labels = labels[indices]
            for i, label in enumerate(labels):
                result += label + ' ' * (max_label_length - len(label))
                for crit_name, imp in self._feature_importances_dict.items():
                    imp = imp[indices]
                    result += '%.2f' % imp[i] + (len(crit_name) ) * ' '
                result += '\n'
        return result

    def linear_fit(self, X, y=None, sample_weight=None, output_weight=None,
                   missing=None, skip_scrub=False):
        """
        Solve the final weighted linear system in basis space.

        This step computes final coefficients and summary statistics such as
        MSE, GCV, RSQ, and GRSQ.
        """
    
        # Format data
        if not skip_scrub:
            X, y, sample_weight, output_weight, missing = self._scrub(
                X, y, sample_weight, output_weight, missing)

        self.coef_ = []
        resid_ = []
        total_weight = 0.
        mse0 = 0.
        for i in range(y.shape[1]):

            # Figure out the weight column
            if sample_weight.shape[1] > 1:
                w = sample_weight[:, i]
            else:
                w = sample_weight[:, 0]

            # Transform into basis space
            B = self.transform(X, missing)  # * w[:, None]
            apply_weights_2d(B, w)

            # Compute total weight
            total_weight += np.sum(w)

            # Apply weights to y
            weighted_y = y.copy()
            weighted_y *= np.sqrt(w[:, np.newaxis])

            # Compute the mse0
            mse0 += np.sum((weighted_y[:, i] -
                            np.average(weighted_y[:, i])) ** 2)

            coef, resid = np.linalg.lstsq(B, weighted_y[:, i], rcond=None)[0:2]
            self.coef_.append(coef)
            # `resid` is a numpy array; don't use it as a boolean (DeprecationWarning).
            if resid.size == 0:
                resid = np.array(
                    [np.sum((np.dot(B, coef) - weighted_y[:, i]) ** 2)])
            resid_.append(resid)
        resid_ = np.array(resid_)
        self.coef_ = np.array(self.coef_)
        # Compute the final mse, gcv, rsq, and grsq (may be different from the
        # pruning scores if the model has been smoothed)
        self.mse_ = np.sum(resid_) / total_weight
        mse0 = mse0 / total_weight
        self.gcv_ = gcv(self.mse_,
                        coef.shape[0], X.shape[0],
                        self.get_penalty())
        gcv0 = gcv(mse0,
                   1, X.shape[0],
                   self.get_penalty())
        if mse0 != 0.:
            self.rsq_ = 1.0 - (self.mse_ / mse0)
        else:
            self.rsq_ = 1.0
        if gcv0 != 0.:
            self.grsq_ = 1.0 - (self.gcv_ / gcv0)
        else:
            self.grsq_ = 1.0

    def predict_deriv(self, X, variables=None, missing=None):
        """
        Predict partial derivatives with respect to selected variables.

        Args:
            X: Input samples.
            variables: Variable indices or names. If omitted, all variables are used.
            missing: Optional missing-value mask.
        """

        # check_is_fitted(self, "basis_")

        if type(variables) in (str, int):
            variables = [variables]
        if variables is None:
            variables_of_interest = list(range(len(self.xlabels_)))
        else:
            variables_of_interest = []
            for var in variables:
                if isinstance(var, int):
                    variables_of_interest.append(var)
                else:
                    variables_of_interest.append(self.xlabels_.index(var))
        X, missing = self._scrub_x(X, missing)
        J = np.zeros(shape=(X.shape[0],
                            len(variables_of_interest),
                            self.coef_.shape[0]))
        b = np.empty(shape=X.shape[0])
        j = np.empty(shape=X.shape[0])
        self.basis_.transform_deriv(
            X, missing, b, j, self.coef_, J, variables_of_interest, True)
        return J

    def score(self, X, y=None, sample_weight=None, output_weight=None,
              missing=None, skip_scrub=False):
        """
        Compute the weighted coefficient of determination on given samples.
        """
        
        # check_is_fitted(self, "basis_")
        if not skip_scrub:
            X, y, sample_weight, output_weight, missing = self._scrub(
                X, y, sample_weight, output_weight, missing)
        if sample_weight.shape[1] == 1 and y.shape[1] > 1:
            sample_weight = np.repeat(sample_weight, y.shape[1], axis=1)
        y_hat = self.predict(X)
        if len(y_hat.shape) == 1:
            y_hat = y_hat[:, None]

        residual = y - y_hat
#         total_weight = np.sum(sample_weight)
        mse = np.sum(sample_weight * (residual ** 2))
        y_avg = np.average(y, weights=sample_weight, axis=0)

        mse0 = np.sum(sample_weight * ((y - y_avg) ** 2))
#         mse0 = np.sum(y_sqr * output_weight) / m
        return 1 - (mse / mse0)

    def score_samples(self, X, y=None, missing=None):
        """
        Return per-sample score values based on relative squared error.
        """
    
        X, y, sample_weight, output_weight, missing = self._scrub(
            X, y, None, None, missing)
        y_hat = self.predict(X, missing=missing)
        residual = 1 - (y - y_hat) ** 2 / y**2
        return residual

    def transform(self, X, missing=None):
        """
        Transform input samples into the fitted basis-function space.
        """

        # check_is_fitted(self, "basis_")
        X, missing = self._scrub_x(X, missing)
        B = np.empty(shape=(X.shape[0], self.basis_.plen()), order='F')
        self.basis_.transform(X, missing, B)
        return B

    def get_penalty(self):
        """Return the active pruning penalty. Defaults to `3.0`."""
        if 'penalty' in self.__dict__ and self.penalty is not None:
            return self.penalty
        else:
            return 3.0


class EarthTrace(object):

    def __init__(self, forward_trace, pruning_trace):
        self.forward_trace = forward_trace
        self.pruning_trace = pruning_trace

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.forward_trace == other.forward_trace and
                self.pruning_trace == other.pruning_trace)

    def __str__(self):
        result = ''
        result += 'Forward Pass\n'
        result += str(self.forward_trace)
        result += '\n'
        result += self.forward_trace.final_str()
        result += '\n\n'
        result += 'Pruning Pass\n'
        result += str(self.pruning_trace)
        result += '\n'
        result += self.pruning_trace.final_str()
        result += '\n'
        return result
