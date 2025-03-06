import numpy as np
from scipy import sparse
from typing import Tuple, Optional, Union

from .core._forward import ForwardPasser
from .core._pruning import PruningPasser
from .core._util import ascii_table, apply_weights_2d, gcv
from .core._types import BOOL
from ..surrogateABC import Surrogate
from ...utility.scalers import Scaler
from ...utility.polynomial_features import PolynomialFeatures

class MARS(Surrogate):

    """
    Multivariate Adaptive Regression Splines(MARS)
    --------------------------------------
    This class is a implementation of MARS from py-earth python library.
    The Multivariate Adaptive Regression Splines(MARS) is a flexible regression method 
    that automatically searches for interactions and non-linear relationships.
    
    The multivariate adaptive regression splines algorithm has two stages.
    First, the forward pass searches for terms in the truncated power spline
    basis that locally minimize the squared error loss of the training set.
    Next, a pruning pass selects a subset of those terms that produces
    a locally minimal generalized cross-validation (GCV) score.  The GCV score
    is not actually based on cross-validation, but rather is meant to
    approximate a true cross-validation score by penalizing model complexity.
    The final result is a set of terms that is nonlinear in the original
    feature space, may include interactions, and is likely to generalize well.
    
    Attributes:
        gcv_ : float
            The generalized cross-validation score of the model.
    
    
    References:
        [1] Friedman, Jerome. Multivariate Adaptive Regression Splines.
            Annals of Statistics. Volume 19, Number 1 (1991), 1-67.
        [2] Fast MARS, Jerome H.Friedman, Technical Report No.110, May 1993.
        [3] Estimating Functions of Mixed Ordinal and Categorical Variables
            Using Adaptive Splines, Jerome H.Friedman, Technical Report
            No.108, June 1991.
        [4] http://www.milbo.org/doc/earth-notes.pdf
        
    endspan_alpha : float, optional, probability between 0 and 1 (default=0.05)
        A parameter controlling the calculation of the endspan
        parameter (below).  The endspan parameter is calculated as
        round(3 - log2(endspan_alpha/n)), where n is the number of features.
        The endspan_alpha parameter represents the probability of a run of
        positive or negative error values on either end of the data vector
        of any feature in the data set.  See equation 45, Friedman, 1991.

    endspan : int, optional (default=-1)
        The number of extreme data values of each feature not eligible
        as knot locations. If endspan is set to -1 (default) then the
        endspan parameter is calculated based on endspan_alpah (above).
        If endspan is set to a positive integer then endspan_alpha is ignored.

    minspan_alpha : float, optional, probability between 0 and 1 (default=0.05)
        A parameter controlling the calculation of the minspan
        parameter (below).  The minspan parameter is calculated as

            (int) -log2(-(1.0/(n*count))*log(1.0-minspan_alpha)) / 2.5

        where n is the number of features and count is the number of points at
        which the parent term is non-zero.  The minspan_alpha parameter
        represents the probability of a run of positive or negative error
        values between adjacent knots separated by minspan intervening
        data points. See equation 43, Friedman, 1991.

    minspan : int, optional (default=-1)
        The minimal number of data points between knots.  If minspan is set
        to -1 (default) then the minspan parameter is calculated based on
        minspan_alpha (above).  If minspan is set to a positive integer then
        minspan_alpha is ignored.

    thresh : float, optional (default=0.001)
        Parameter used when evaluating stopping conditions for the forward
        pass. If either RSQ > 1 - thresh or if RSQ increases by less than
        thresh for a forward pass iteration then the forward pass is
        terminated.

    zero_tol : float, optional (default=1e-12)
        Used when determining whether a floating point number is zero during
        the  forward pass.  This is important in determining linear dependence
        and in the fast update procedure.  There should normally be no reason
        to change  zero_tol from its default. However, if nans are showing up
        during the forward pass or the forward pass seems to be terminating
        unexpectedly, consider adjusting zero_tol.

    min_search_points : int, optional (default=100)
        Used to calculate check_every (below).  The minimum samples necessary
        for check_every to be greater than 1.  The check_every parameter
        is calculated as

             (int) m / min_search_points

        if m > min_search_points, where m is the number of samples in the
        training set.  If m <= min_search_points then check_every is set to 1.

    check_every : int, optional (default=-1)
        If check_every > 0, only one of every check_every sorted data points
        is considered as a candidate knot.  If check_every is set to -1 then
        the check_every parameter is calculated based on
        min_search_points (above).

    allow_linear : bool, optional (default=True)
        If True, the forward pass will check the GCV of each new pair of terms
        and, if it's not an improvement on a single term with no knot (called a
        linear term, although it may actually be a product of a linear term
        with some other parent term), then only that single, knotless term will
        be used. If False, that behavior is disabled and all terms will have
        knots except those with variables specified by the linvars argument
        (see the fit method).

    use_fast : bool, optional (default=False)
        if True, use the approximation procedure defined in [2] to speed up the
        forward pass. The procedure uses two hyper-parameters : fast_K
        and fast_h. Check below for more details.

    fast_K : int, optional (default=5)
        Only used if use_fast is True. As defined in [2], section 3.0, it
        defines the maximum number of basis functions to look at when
        we search for a parent, that is we look at only the fast_K top
        terms ranked by the mean squared error of the model the last time
        the term was chosen as a parent. The smaller fast_K is, the more
        gains in speed we get but the more approximate is the result.
        If fast_K is the maximum number of terms and fast_h is 1,
        the behavior is the same as in the normal case
        (when use_fast is False).

    fast_h : int, optional (default=1)
        Only used if use_fast is True. As defined in [2], section 4.0, it
        determines the number of iterations before repassing through all
        the variables when searching for the variable to use for a
        given parent term. Before reaching fast_h number of iterations
        only the last chosen variable for the parent term is used. The
        bigger fast_h is, the more speed gains we get, but the result
        is more approximate.

    smooth : bool, optional (default=False)
        If True, the model will be smoothed such that it has continuous first
        derivatives.
        For details, see section 3.7, Friedman, 1991.

    enable_pruning : bool, optional(default=True)
        If False, the pruning pass will be skipped.

    feature_importance_type: string or list of strings, optional (default=None)
        Specify which kind of feature importance criteria to compute.
        Currently three criteria are supported : 'gcv', 'rss' and 'nb_subsets'.
        By default (when it is None), no feature importance is computed.
        Feature importance is a measure of the effect of the features
        on the outputs. For each feature, the values go from
        0 to 1 and sum up to 1. A high value means the feature have in average
        (over the population) a large effect on the outputs.
        See [4], section 12.3 for more information about the criteria.

    verbose : int, optional(default=0)
        If verbose >= 1, print out progress information during fitting.  If
        verbose >= 2, also print out information on numerical difficulties
        if encountered during fitting. If verbose >= 3, print even more
        information that is probably only useful to the developers of py-earth.
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

    def __init__(self, scalers: Tuple[Optional[Scaler], Optional[Scaler]] = (None, None),
                 polyFeature: PolynomialFeatures = None, 
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
        '''
        Initialize the MARS surrogate model.
        
        :param max_terms: int, optional 
            default = min(2 * n + m // 10, 400)), where n is the number of features and m is the number of rows
            The maximum number of terms generated by the forward pass. 
        :param max_degree: int, optional (default=1)
            The maximum degree of terms generated by the forward pass.
        :param penalty: float, optional (default=3.0)
            A smoothing parameter used to calculate GCV and GRSQ.
            Used during the pruning pass and to determine whether to add a hinge
            or linear basis function during the forward pass.
        :param endspan_alpha: float, optional (default=0.05)
            A parameter controlling the calculation of the endspan
            parameter (below).  The endspan parameter is calculated as
            round(3 - log2(endspan_alpha/n)), where n is the number of features.
            The endspan_alpha parameter represents the probability of a run of
            positive or negative error values on either end of the data vector
            of any feature in the data set.  See equation 45, Friedman, 1991.
        :param endspan: int, optional (default=-1)
            The number of extreme data values of each feature not eligible
            as knot locations. If endspan is set to -1 (default) then the
            endspan parameter is calculated based on endspan_alpah (above).
            If endspan is set to a positive integer then endspan_alpha is ignored.
        :param minspan_alpha: float, optional (default=0.05)
            A parameter controlling the calculation of the minspan
            parameter (below).  The minspan parameter is calculated as
            (int) -log2(-(1.0/(n*count))*log(1.0-minspan_alpha)) / 2.5
            where n is the number of features and count is the number of points at
            which the parent term is non-zero.  The minspan_alpha parameter
            represents the probability of a run of positive or negative error
            values between adjacent knots separated by minspan intervening
            data points. See equation 43, Friedman, 1991.
        :param minspan: int, optional (default=-1)
            The minimal number of data points between knots.  If minspan is set
            to -1 (default) then the minspan parameter is calculated based on
            minspan_alpha (above).  If minspan is set to a positive integer then
            minspan_alpha is ignored.
        :param thresh: float, optional (default=0.001)
            Parameter used when evaluating stopping conditions for the forward
            pass. If either RSQ > 1 - thresh or if RSQ increases by less than
            thresh for a forward pass iteration then the forward pass is
            terminated.
        :param zero_tol: float, optional (default=1e-12)
            Used when determining whether a floating point number is zero during
            the  forward pass.  This is important in determining linear dependence
            and in the fast update procedure.  There should normally be no reason
            to change  zero_tol from its default. However, if nans are showing up
            during the forward pass or the forward pass seems to be terminating
            unexpectedly, consider adjusting zero_tol.
        :param min_search_points: int, optional (default=100)
            Used to calculate check_every (below).  The minimum samples necessary
            for check_every to be greater than 1.  The check_every parameter
            is calculated as
             (int) m / min_search_points
            if m > min_search_points, where m is the number of samples in the
            training set.  If m <= min_search_points then check_every is set to 1.
        :param check_every: int, optional (default=-1)
            If check_every > 0, only one of every check_every sorted data points
            is considered as a candidate knot.  If check_every is set to -1 then
            the check_every parameter is calculated based on
            min_search_points (above).
        :param allow_linear: bool, optional (default=True)
            If True, the forward pass will check the GCV of each new pair of terms
            and, if it's not an improvement on a single term with no knot (called a
            linear term, although it may actually be a product of a linear term
            with some other parent term), then only that single, knotless term will
            be used. If False, that behavior is disabled and all terms will have
            knots except those with variables specified by the linvars argument
            (see the fit method).
        :param use_fast: bool, optional (default=False)
            if True, use the approximation procedure defined in [2] to speed up the
            forward pass. The procedure uses two hyper-parameters : fast_K
            and fast_h. Check below for more details.
        :param fast_K: int, optional (default=5)
            Only used if use_fast is True. As defined in [2], section 3.0, it
            defines the maximum number of basis functions to look at when
            we search for a parent, that is we look at only the fast_K top
            terms ranked by the mean squared error of the model the last time
            the term was chosen as a parent. The smaller fast_K is, the more
            gains in speed we get but the more approximate is the result.
            If fast_K is the maximum number of terms and fast_h is 1,
            the behavior is the same as in the normal case
            (when use_fast is False).
        :param fast_h: int, optional (default=1)
            Only used if use_fast is True. As defined in [2], section 4.0, it
            determines the number of iterations before repassing through all
            the variables when searching for the variable to use for a
            given parent term. Before reaching fast_h number of iterations
            only the last chosen variable for the parent term is used. The
            bigger fast_h is, the more speed gains we get, but the result
            is more approximate.
        :param smooth: bool, optional (default=False)
            If True, the model will be smoothed such that it has continuous first
            derivatives.
            For details, see section 3.7, Friedman, 1991.
        :param enable_pruning: bool, optional(default=True)
            If False, the pruning pass will be skipped.
        :param feature_importance_type: string or list of strings, optional (default=None)
            Specify which kind of feature importance criteria to compute.
            Currently three criteria are supported : 'gcv', 'rss' and 'nb_subsets'.
            By default (when it is None), no feature importance is computed.
            Feature importance is a measure of the effect of the features
            on the outputs. For each feature, the values go from
            0 to 1 and sum up to 1. A high value means the feature have in average
            (over the population) a large effect on the outputs.
            See [4], section 12.3 for more information about the criteria.            
        '''
        
        super().__init__(scalers, polyFeature)
        
        
        allow_missing = False
        verbose = 0
        
        self.setting.setPara("max_terms", max_terms, max_terms_attr)
        self.setting.setPara("max_degree", max_degree, max_degree_attr)
        self.setting.setPara("penalty", penalty, penalty_attr)
        
        self.setting.setPara("endspan_alpha", endspan_alpha)
        self.setting.setPara("endspan", endspan)
        self.setting.setPara("minspan_alpha", minspan_alpha)
        self.setting.setPara("minspan", minspan)
        self.setting.setPara("thresh", thresh)
        self.setting.setPara("zero_tol", zero_tol)
        self.setting.setPara("min_search_points", min_search_points)
        self.setting.setPara("check_every", check_every)
        self.setting.setPara("allow_linear", allow_linear)
        self.setting.setPara("use_fast", use_fast)
        self.setting.setPara("fast_K", fast_K)
        self.setting.setPara("fast_h", fast_h)
        self.setting.setPara("smooth", smooth)
        self.setting.setPara("enable_pruning", enable_pruning)
        self.setting.setPara("feature_importance_type", feature_importance_type)
        self.setting.setPara("verbose", verbose)
        self.setting.setPara("allow_missing", allow_missing)
        
#-------------------------Public Function---------------------------#
    def fit(self, xTrain: np.ndarray, yTrain: np.ndarray):
        
        xTrain, yTrain=self.__check_and_scale__(xTrain, yTrain)
        
        #indicate label for each dimension
        self.xlabels_ = self._scrape_labels(xTrain)
        xTrain, yTrain, sample_weight, output_weight, missing = self._scrub(
            xTrain, yTrain, None, None, None)
        #forward
        self.forward_pass(xTrain, yTrain,
                          sample_weight, output_weight, missing,
                          self.xlabels_, [], skip_scrub=True)
        #pruning
        if self.setting.getVals("enable_pruning") is True:
            self.pruning_pass(xTrain, yTrain,
                              sample_weight, output_weight, missing,
                              skip_scrub=True)
        if self.setting.getVals("smooth"):
            self.basis_ = self.basis_.smooth(xTrain)
        self.linear_fit(xTrain, yTrain, sample_weight, output_weight, missing,
                        skip_scrub=True)
        return self
    
    def predict(self, xPredict: np.ndarray):
        
        xPredict = self.__X_transform__(xPredict)
        
        X, missing = self._scrub_x(xPredict, None)
        B = self.transform(X, missing)
        y = np.dot(B, self.coef_.T)
        
        return self.__Y_inverse_transform__(y)

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
                result[name] = setting.parVal[name]
            elif name in setting.parCon.keys():
                result[name] = setting.parCon[name]
        return result

    def _pull_pruning_args(self, setting):
        '''
        Pull named arguments relevant to the pruning pass.
        '''
        result = {}
        for name in self.pruning_pass_arg_names:
            if name in setting.parVal.keys():
                result[name] = setting.parVal[name]
            elif name in setting.parCon.keys():
                result[name] = setting.parCon[name]
        return result

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
            if not self.setting.getVals("allow_missing"):
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
        '''Return information about the forward pass.'''
        try:
            return self.forward_pass_record_
        except AttributeError:
            return None

    def pruning_trace(self):
        '''Return information about the pruning pass.'''
        try:
            return self.pruning_pass_record_
        except AttributeError:
            return None

    def trace(self):
        '''Return information about the forward and pruning passes.'''
        return EarthTrace(self.forward_trace(), self.pruning_trace())

    def summary(self):
        '''Return a string describing the model.'''
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
            if not resid:
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
    
        X, y, sample_weight, output_weight, missing = self._scrub(
            X, y, None, None, missing)
        y_hat = self.predict(X, missing=missing)
        residual = 1 - (y - y_hat) ** 2 / y**2
        return residual

    def transform(self, X, missing=None):

        # check_is_fitted(self, "basis_")
        X, missing = self._scrub_x(X, missing)
        B = np.empty(shape=(X.shape[0], self.basis_.plen()), order='F')
        self.basis_.transform(X, missing, B)
        return B

    def get_penalty(self):
        '''Get the penalty parameter being used.  Default is 3.'''
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
