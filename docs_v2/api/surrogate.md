# Surrogate API

SVR returns an approximate model with a Python `RuntimeWarning` on reaching `maxIter`; it no longer prints the C++ terminal message. `fitState["solver"]` records `iterations`, `maxIterations`, and `iterationLimitReached`. Reaching the limit does not establish convergence; stopping below it does not guarantee predictive accuracy. AutoTuner records `iterations`, `max_iterations`, and `iteration_limit_reached` separately in each `candidates[i]["fit"]["solver"]` and `final_refit["solver"]`. A `finished` status means the call completed. Approximate candidates still participate in validation-based selection, and warnings do not automatically increase the budget. Promoting warnings to errors preserves the usual fit invalidation behavior.


GPR `C` is the observation noise variance / regularization added to the kernel diagonal, not a standard deviation. Its initial value remains `1e-9`; the default `C_attr` searches `[1e-12, 1]` in log coordinates to allow nearly noiseless and noisy fits. `C_attr=None` fixes C; explicit bounds take precedence. C uses the variance units of the preprocessed target, including any output scaler. The default upper bound cannot cover arbitrary raw output units; configure an output scaler or custom bounds for other scales. A wider search does not force higher noise or guarantee accuracy or uncertainty calibration.


GPR RBF, Matern and RationalQuadratic kernels search length scales over `[0.01, 1e5]` by default in log coordinates. Lengths use the preprocessed input units; configure an input scaler or explicit `length_attr` for other scales. `length_attr=None` still fixes the length. No search range guarantees accuracy on every problem. Standalone MARS now defaults to `max_degree=2` to allow pairwise interactions; explicit `max_degree=1` retains an additive model. Degree two can increase fitting cost.

SVR `fit()` uses the current parameters without automatic hyperparameter search. Use `AutoTuner.gridTune`/`optTune` to select parameters on an internal validation split and refit all training rows, then evaluate on separate test data. C, epsilon and gamma are log parameters by default, so explicit grids use encoded values, for example `{"C": np.log([1, 100]), "epsilon": np.log([0.001, 0.01]), "gamma": np.log([1, 10, 100])}`. This is a starting grid, not a universal optimum. Consider scalers and parameter ranges when input or target units change.

GPR/KRG standard deviations describe uncertainty under the fitted model assumptions, not guaranteed actual prediction errors. Misspecification, sparse data and extrapolation can produce overconfidence. The wider search resolves the reproduced short-scale GPR failure; it does not establish calibration on arbitrary data.


MARS can be fitted repeatedly with different input dimensions. Each fit clears the previous learned basis, coefficients and traces; failed fitting invalidates the model until a later successful fit. `plot_surrogate` pads default axes using the combined true/predicted data span, including negative and constant data; explicit `ylim` takes precedence.

MARS transform accepts already preprocessed inputs. Transform, prediction, scoring, and summary methods require valid fitted state; forward_trace/pruning_trace return None when invalid. predict accepts keyword-only missing, which score/score_samples preserve. Missing values are disabled by default; enable them with model.setting.set("allow_missing", True). Without input preprocessing, NaNs or an input-shaped boolean mask are supported; missing predictors combined with an input scaler or PolyFeature raise NotImplementedError. score_samples retains the per-sample, per-output definition 1-(y-prediction)²/y², including NumPy division semantics at zero targets. gridTune rejects unknown or non-tunable names while accepting parameters activated by selected structural configurations; validation performs no additional model fits.

AutoTuner gridTune/optTune invalidate the model on any exception or interruption, including preprocessing, parameter application, search, and final full-data refitting. The original exception propagates; refit successfully before prediction. Parameters and scalers are not rolled back. MultiSurrogate retains supplied model references but requires distinct instances per output, checked on construction, append, fit, and predict. MARS.predict_deriv accepts raw inputs and returns derivatives in original input/output units with shape (n_samples, n_selected_variables, 1). It supports StandardScaler/MinMaxScaler and no preprocessing; PolyFeature and other non-affine scalers raise NotImplementedError.

`rank_score` averages per-output Kendall tau-b, assigns zero to constant columns, and requires at least two samples. `MultiSurrogate.rng` seeds independent child streams. AutoTuner records numerical linear-algebra/arithmetic failures and nonfinite predictions/scores in `candidateFailures` (candidate_index, error_type, message), reset per tuning call; programming errors propagate without being printed and swallowed.

Surrogate models copy supplied input/output scalers, so fitting one model does not refit another model’s scaler. A failed public `fit()` invalidates fitted state: prediction raises until a subsequent successful fit. MSE/R²/NSE accept `(n,)` as single-output `(n, 1)`, require matching finite nonempty arrays, and reject implicit broadcasting. AutoTuner requires at least two validation samples and finite nonzero total validation variation for its aggregate R² score; increase `ratio` when necessary. If every candidate fails or scores non-finitely, tuning raises instead of returning an arbitrary candidate. Direct R²/NSE calls retain their non-finite result for constant targets.

For GPR/KRG, `nRestartTimes` counts additional searches: `0` means one search. The default `None` resolves to 4 restarts (5 searches total) for local optimizers (Boxmin/LBFGSB/MP), and retains 1 restart for EA. Local optimization starts from the current configured parameters (including fitted values on a repeated fit), then samples uniformly within optimization-coordinate bounds, hence in log space for log parameters. Set `model.rng = np.random.default_rng(42)` for reproducibility with identical data, initial parameters and RNG state. Selection uses finite objectives recomputed at returned points; all-invalid candidates raise an error. More restarts do not guarantee lower prediction error.

`LBFGSB` returns the best finite evaluated point (including finite-difference probes) and its matching objective; this does not imply convergence. Its `lastResult` retains the last raw SciPy result, including `success/status/message`; its `x/fun` may differ from the wrapper return. Objectives must be deterministic; no finite candidate raises an error. Pass `LBFGSB(options={"maxls": 50})` explicitly to increase the line-search step limit, without a universal improvement guarantee.

## `UQPyL.surrogate`

The `surrogate` module trains predictive models for expensive simulations, objectives, or intermediate response surfaces.

### Import

```python
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate import MinMaxScaler, StandardScaler, KFold, AutoTuner
```

### Public Objects

Top-level objects:

| Object | Role |
|---|---|
| `SurrogateABC` | Base class for surrogate models. |
| `MultiSurrogate` | Container for one surrogate per output column. |
| `AutoTuner` | Hyper-parameter tuning helper. |
| `PolyFeature` | Polynomial feature expansion. |
| `KFold` | K-fold index splitter. |
| `RandSelect` | Random train/test splitter. |
| `Scaler` | Base scaler interface. |
| `MinMaxScaler` | Min-max scaler. |
| `StandardScaler` | Standard scaler. |

Model subpackages:

| Subpackage | Import path | Main objects |
|---|---|---|
| RBF | `UQPyL.surrogate.rbf` | `RBF`, `Cubic`, `Linear`, `Multiquadric`, `ThinPlateSpline`, `Gaussian` |
| Gaussian process | `UQPyL.surrogate.gp` | `GPR` |
| Kriging | `UQPyL.surrogate.kriging` | `KRG` |
| Regression | `UQPyL.surrogate.regression` | `LinearRegression`, `PolynomialRegression` |
| MARS | `UQPyL.surrogate.mars` | `MARS` |
| SVR | `UQPyL.surrogate.svr` | `SVR` |

`mars` and `svr` may be unavailable when optional compiled dependencies are not installed.

## Fit and Predict

All surrogate models follow the same high-level protocol:

```python
model.fit(xTrain, yTrain)
yPred = model.predict(xPred)
```

Each surrogate fits one output: `yTrain` must have shape `(nTrain,)` or
`(nTrain,1)`, normalized to two dimensions internally. Mean, standard-deviation,
and variance predictions have shape `(nPred,1)`. Multiple output columns are
rejected before preprocessing, prepared fitting, and AutoTuner search. Use
MultiSurrogate for multiple outputs and tune its child models individually.
Scalers and standalone scoring utilities may still process multiple columns.
Raw `fit`/`prepareTrainingData` accept vectors; prepared `fitModel`/`fitHyper`
consume the resulting `(nTrain,1)` matrix.

Example:

```python
import numpy as np

from UQPyL.surrogate.rbf import RBF


X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
Y = np.sin(2 * np.pi * X)

model = RBF()
model.fit(X, Y)

pred = model.predict([[0.25], [0.75]])
print(pred)
```

Uncertainty output uses the common flags:

```python
mean = model.predict(X)
mean, std = model.predict(X, returnStd=True)
mean, var = model.predict(X, returnVar=True)
```

Only models with `supportsUncertainty=True` support `returnStd` or `returnVar`.

## `SurrogateABC`

Base class for surrogate models.

```python
SurrogateABC(scalers=(None, None), polyFeature=None)
```

| Parameter | Meaning |
|---|---|
| `scalers` | Pair of optional `(xScaler, yScaler)`. |
| `polyFeature` | Optional `PolyFeature` applied after input scaling. |

Common methods:

| Method | Returns | Meaning |
|---|---|---|
| `fit(xTrain, yTrain)` | model | Prepare data, fit hyper-parameters, and fit the model. |
| `predict(xPred, returnStd=False, returnVar=False)` | `np.ndarray` or tuple | Predict output values, optionally with uncertainty. |
| `prepareTrainingData(xTrain, yTrain)` | `(X, Y)` | Validate, scale, and transform training data. |
| `storeTrainingData(xTrain, yTrain)` | model | Store prepared training data. |
| `requireFitted(*stateKeys)` | model | Raise if the model is not fitted. |
| `getParaList()` | `list` | Return active tunable parameter names. |
| `applyParameterValues(paraList, values, ignoreInactive=True, *, paraInfos=None)` | model | Apply a flat candidate in encoded parameter coordinates; optional slices come from `Setting.getParaInfos`. |
| `getParameterValues(*args, ignoreInactive=False)` | value or tuple | Return current parameter values. |

## `MultiSurrogate`

Container for multi-output surrogate prediction.

```python
MultiSurrogate(n_surrogates, models_list=None)
```

| Method | Returns | Meaning |
|---|---|---|
| `append(model)` | `None` | Append one `SurrogateABC` model. |
| `fit(trainX, trainY)` | container | Fit one model per output column. |
| `predict(testX, returnStd=False, returnVar=False)` | array or tuple | Stack means and optional marginal uncertainty by output. |
| `predict_deriv(testX, variables=None, missing=None)` | array | Stack supported derivatives on the last axis. |

`trainY.shape[1]` and `len(models_list)` must match `n_surrogates`.
Each child receives a `(nTrain,1)` target. Inputs and outputs must have matching
sample counts. Failed or interrupted fitting invalidates all children; refit
successfully before predicting. Every child prediction must retain one output
column and the correct sample count.

Means and uncertainties have shape `(nPred,n_surrogates)`; derivatives have
shape `(nPred,nVariables,n_surrogates)`. `supportsUncertainty` is true only when
all children support it. Derivative aggregation likewise requires every child
to provide `predict_deriv`. Each output uses its own scaling and model; returned
variances are marginal variances, with no cross-output covariance.

Standard deviations are restored directly, without first forming original-unit
variances. An unrepresentable variance emits `RuntimeWarning` and returns zero
on underflow or infinity on overflow; request `returnStd=True` when the standard
deviation remains representable. Custom output scalers must implement the
corresponding `inverse_transform_std` or `inverse_transform_var` method.

Numerical and input contracts:

- RBF normally solves the complete LU system without truncating polynomial
  constraints. Singular systems emit `RuntimeWarning` and use a constrained
  least-squares approximation. `fitState["linearSolve"]` records the method
  (`lu` or `constrained_lstsq`), degeneracy flags, rank, relative training
  residual in prepared target units, and constraint residual. Conflicting
  repeated observations cannot be interpolated exactly; deficient trends do
  not determine unique extrapolation. If the fallback fails or produces
  nonfinite results, fitting stops and previous fitted state is invalidated.
- Lasso copies integer or mixed-dtype inputs to a common floating working dtype
  and preserves the supplied arrays.
- nu-SVR searches nu in `[1e-5,1]` by default. Parameters and custom search
  bounds must satisfy `0 < nu <= 1`.
- GPR requires nonempty finite training data with matching sample counts and a
  finite nonnegative scalar C. Logarithmic C search bounds must be positive.
  Invalid inputs are rejected before backend fitting/search; failed refits
  invalidate previous fitted state.
- StandardScaler computes sample standard deviations (`ddof=1`) in binary
  scaled units, and keeps source and target origins separate for constant
  columns and nondefault target centers.
- R²/NSE use safely scaled squared sums while retaining relative output weights.
  Scores for constant targets remain undefined: NaN for exact predictions,
  otherwise negative infinity, with NumPy warnings controlled by `np.errstate`. AutoTuner
  checks constancy directly, rather than rejecting valid targets because their
  original-unit squared variation underflows or overflows.

## Models

### `RBF`

Radial basis function surrogate.

```python
RBF(
    scalers=(None, None),
    polyFeature=None,
    kernel=Cubic(),
    C_smooth=0.0,
    C_smooth_attr={...},
)
```

| Parameter | Meaning |
|---|---|
| `kernel` | RBF kernel object. |
| `C_smooth` | Finite, nonnegative scalar smoothing strength; zero disables smoothing. |
| `C_smooth_attr` | Tuning metadata for `C_smooth`. |

Additional methods:

| Method | Meaning |
|---|---|
| `setKernel(kernel)` | Replace the active kernel. |
| `setKernelChoices(kernels)` | Register tunable kernel choices. |

Available RBF kernels:

| Kernel |
|---|
| `Cubic` |
| `Linear` |
| `Multiquadric` |
| `ThinPlateSpline` |
| `Gaussian` |

Smoothing modifies only the diagonal of the training kernel block, preserving
the polynomial tail and its constraints. With the existing kernel definitions,
`Cubic`, `ThinPlateSpline`, and `Gaussian` use `+C_smooth`; `Linear` and
`Multiquadric` use `-C_smooth` because their positive-valued kernels are
conditionally negative definite. This follows the smoothing equations in
[SciPy's RBFInterpolator documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html),
after accounting for its opposite signs for Linear and Multiquadric.
For well-posed training data, smoothing preserves affine trends for Cubic and
ThinPlateSpline, and constants for Linear and Multiquadric. Gaussian has no
polynomial tail. The smoothing strength applies in the prepared training space.

### `GPR`

Gaussian process regression surrogate.

```python
GPR(
    scalers=(None, None),
    polyFeature=None,
    kernel=RBFKernel(),
    optimizer="Boxmin",
    nRestartTimes=4,
    C=1e-9,
    C_attr={...},
)
```

| Parameter | Meaning |
|---|---|
| `kernel` | Gaussian process kernel. |
| `optimizer` | Hyper-parameter optimizer. |
| `nRestartTimes` | Number of hyper-parameter optimization restarts. |
| `C` | Numerical regularization parameter. |
| `C_attr` | Tuning metadata for `C`. |

Additional methods:

| Method | Meaning |
|---|---|
| `setKernel(kernel)` | Replace the active kernel. |
| `setKernelChoices(kernels)` | Register tunable kernel choices. |

`GPR` supports uncertainty output. Internal MP and EA optimizers minimize the
negative log marginal likelihood. `fitState["objective"]` records this same
quantity on the prepared training data; lower values are better, and the value
can be negative.

`GPR` and `KRG` default to `Boxmin`. It maps finite parameter bounds to a
positive internal interval `[1, 2]` for multiplicative search, then evaluates
and returns parameters in the supplied coordinates (including log coordinates
when configured). Evaluation points stay within those bounds; fixed parameters
remain fixed and out-of-bounds initial points are clipped. This internal mapping
does not change the model's input scaling. `Boxmin` and `LBFGSB` use local random
generators, so their initialization does not change NumPy's global random state.

### `KRG`

Kriging surrogate.

```python
KRG(
    scalers=(None, None),
    polyFeature=None,
    kernel=Guass(),
    regression="poly0",
    optimizer="Boxmin",
    nRestartTimes=4,
)
```

| Parameter | Meaning |
|---|---|
| `kernel` | Kriging correlation kernel. |
| `regression` | Regression trend. One of `"poly0"`, `"poly1"`, or `"poly2"`. |
| `optimizer` | Hyper-parameter optimizer. |
| `nRestartTimes` | Number of hyper-parameter optimization restarts. |

Additional methods:

| Method | Meaning |
|---|---|
| `setKernel(kernel)` | Replace the active kernel. |
| `setKernelChoices(kernels)` | Register tunable kernel choices. |

`KRG` supports uncertainty output.

### `LinearRegression`

Linear regression surrogate.

```python
LinearRegression(
    scalers=(None, None),
    polyFeature=None,
    lossType="Origin",
    fitIntercept=True,
    C=0.1,
    C_attr={...},
    maxIter=100,
    maxEpoch=500000.0,
    tolerance=0.001,
    p0=10,
)
```

| Parameter | Meaning |
|---|---|
| `lossType` | Regression loss. One of `"Origin"`, `"Ridge"`, or `"Lasso"`. |
| `fitIntercept` | Whether to fit an intercept term. |
| `C` | Regularization parameter. |
| `maxIter` | Maximum outer iterations. |
| `maxEpoch` | Maximum Lasso epochs. |
| `tolerance` | Convergence tolerance. |
| `p0` | Lasso working-set parameter. |

Lasso centers private working arrays, leaving supplied and stored training data
unchanged during fitting. This also applies to `PolynomialRegression` with
`lossType="Lasso"`, allowing prepared data to be reused across tuning candidates.

Origin, Ridge, and Lasso all follow the shared single-output contract. Use
MultiSurrogate with one independent regression instance per output column.

### `PolynomialRegression`

Polynomial regression surrogate.

```python
PolynomialRegression(
    scalers=(None, None),
    degree=2,
    degree_attr={...},
    onlyInteraction=False,
    lossType="Origin",
    fitIntercept=True,
    C=0.1,
    C_attr={...},
    maxIter=100,
    maxEpoch=500000.0,
    tolerance=0.001,
    p0=10,
)
```

| Parameter | Meaning |
|---|---|
| `degree` | Polynomial degree. |
| `onlyInteraction` | Whether to use interaction-only polynomial terms. |
| `lossType` | Regression loss. One of `"Origin"`, `"Ridge"`, or `"Lasso"`. |

### `MARS`

Multivariate adaptive regression splines surrogate.

```python
MARS(
    scalers=(None, None),
    polyFeature=None,
    max_terms=400,
    max_degree=2,
    penalty=3.0,
    endspan_alpha=0.05,
    endspan=-1,
    minspan_alpha=0.05,
    minspan=-1,
    thresh=0.001,
    zero_tol=1e-12,
    min_search_points=100,
    check_every=-1,
    allow_linear=True,
    use_fast=False,
    fast_K=5,
    fast_h=1,
    smooth=False,
    enable_pruning=True,
    feature_importance_type="gcv",
)
```

| Parameter | Meaning |
|---|---|
| `max_terms` | Maximum number of basis terms. |
| `max_degree` | Maximum interaction degree. |
| `penalty` | Generalized cross-validation penalty. |
| `enable_pruning` | Whether to prune basis terms. |
| `feature_importance_type` | Feature-importance calculation mode. |

### `SVR`

Support vector regression surrogate.

```python
SVR(
    scalers=(None, None),
    polyFeature=None,
    symbol="epsilon-SVR",
    kernel="rbf",
    nu=0.5,
    C=0.1,
    epsilon=0.1,
    gamma=1.0,
    coe0=0.1,
    degree=3,
    maxIter=100000.0,
    eps=0.001,
)
```

| Parameter | Meaning |
|---|---|
| `symbol` | SVR type. One of `"epsilon-SVR"` or `"nu-SVR"`. |
| `kernel` | Kernel. One of `"linear"`, `"rbf"`, `"sigmoid"`, or `"polynomial"`. |
| `nu` | Nu-SVR parameter. |
| `C` | Regularization parameter. |
| `epsilon` | Epsilon-SVR tube width. |
| `gamma` | Kernel gamma. |
| `coe0` | Kernel coefficient. |
| `degree` | Polynomial kernel degree. |
| `maxIter` | Maximum solver iterations. |
| `eps` | Solver tolerance. |

## Scaling and Features

### `MinMaxScaler`

```python
MinMaxScaler(min_=0, max_=1)
```

Maps each feature to `[min_, max_]`.

### `StandardScaler`

```python
StandardScaler(muX=0, sitaX=1)
```

Maps each feature to the requested mean and standard deviation scale.

Scaler methods:

| Method | Meaning |
|---|---|
| `fit(trainX)` | Fit scaler statistics. |
| `transform(trainX)` | Transform data. |
| `fit_transform(trainX)` | Fit and transform data. |
| `inverse_transform(trainX)` | Reverse the transformation. |

### `PolyFeature`

```python
PolyFeature(degree=2, includeBias=False, onlyInteraction=False)
```

| Method | Returns | Meaning |
|---|---|---|
| `transform(trainX)` | `np.ndarray` | Expand polynomial features. |

## Splitting and Metrics

### `KFold`

```python
KFold(n_splits=5)
```

| Method | Returns | Meaning |
|---|---|---|
| `split(X, mode="full")` | `(train, test)` | Return fold indices. `mode` is `"full"` or `"single"`. |

### `RandSelect`

```python
RandSelect(pTest=5)
```

| Method | Returns | Meaning |
|---|---|---|
| `split(X)` | `(train, test)` | Return one random train/test split. `pTest` is a percentage. |

Metrics:

| Function | Meaning |
|---|---|
| `r_square(true_Y, pre_Y)` | R-squared score. |
| `nse(true_Y, pre_Y)` | Nash-Sutcliffe efficiency. |
| `mse(true_Y, pre_Y)` | Mean squared error. |
| `rank_score(true_Y, pre_Y)` | Kendall-style rank score. |
| `sort_score(true_Y, pre_Y)` | Sorted-index distance score. |

## `AutoTuner`

Hyper-parameter tuning helper for surrogate models.

```python
AutoTuner(model, optimizer=None)
```

| Method | Returns | Meaning |
|---|---|---|
| `optTune(xData, yData, paraList=None, ratio=10, owner=None, tuneMode="separate", *, seed=None, rng=None)` | `(bestParams, bestScore)` | Maximize validation R² with an optimizer. |
| `gridTune(xData, yData, paraGrid=None, ratio=10, owner=None, tuneMode="separate", *, seed=None, rng=None)` | `(bestParams, bestScore)` | Maximize validation R² over a parameter grid. |

`paraList` defaults to registered tunable parameters, optionally filtered by owner.
`optTune` fixes the parameter slices and bounds at the start of a search; each
vector occupies its full number of coordinates. For kernel choices, shared
parameters must have matching dimensions, bounds, types and log settings.
Incompatible spaces raise an error; configure compatible choices or tune them separately.

`paraGrid` maps names to candidate values in the same encoded coordinates:
use natural logs for log parameters and bin coordinates for numeric choices.
A vector is one candidate, for example
`paraGrid={"l": [np.log([0.2, 3.0]), np.log([2.0, 0.2])]}` for a two-dimensional
length scale. The grid iterates over combinations without flattening these vectors.
With `paraGrid=None`, it evaluates the current parameter values once.

Structural parameters are applied before numeric parameters. Parameters absent
from the selected kernel are skipped, and their returned best values are `None`.
Use `tuneMode="joint"` to fit exactly the candidate parameters; `"separate"`
also runs the model's internal tuning. Returned numeric parameters are decoded
values. The selected model is refitted on all supplied data; the reported score
is its validation score from the search split.


GPR/KRG mean-only prediction skips uncertainty solves. GPR obtains self-kernel diagonals directly for built-in kernels, without allocating a prediction-by-prediction matrix; custom GP kernels can override `diag(X)` or use the bounded-block fallback. GPR/KRG/RBF share template installation and parameter merging while retaining their own mathematical kernel families. Unimplemented surrogate ensemble placeholders have been removed.

## Tuning reports and explicit validation splits

`gridTune` and `optTune` accept mutually exclusive keyword-only `splitIndices=(trainIdx, validationIdx)` and `splitter`.
Index arrays must be nonempty, one-dimensional, integral, unique, in bounds, and disjoint. Unused rows are allowed, for example a temporal gap.
A splitter is a callable or an object exposing `split(X)` that returns one index pair. It receives a copy of X and an independent `seed` or `rng` when its signature supports that argument.
Without either option, `ratio` remains the random **validation percentage**. Explicit splits ignore ratio. Actual indices and seeds are recorded in `lastSplit` and the report.
One tuning call uses one split: after `trainFolds, validationFolds = KFold(...).split(X)`, a selected `(trainFolds[0], validationFolds[0])` pair is supported; automatic multi-fold aggregation is not.
Preprocessing is fitted on training rows during selection. Final refitting uses **all supplied rows**, including unused rows. Keep external test data outside the tuning input.

`joint` calls `fitModel` on each exact candidate; default `separate` calls `fitHyper`, allowing internal optimization to change the proposed parameters.
The returned parameters belong to the final full-data refit; the returned score belongs to validation during selection, not an external test of the final model.
`getReport()` returns an independent report with encoded candidates, actual parameters before/after fitting, validation scores, final refit parameters, fit counts, tracked objective calls, timings, and failure details.
`candidate_encoded` follows Setting coordinates, including log encoding; fitted parameter snapshots contain actual values.
Only instrumentable `_objfunc` calls contribute to `tracked_objective_evaluations`; unsupported per-fit counts are `None`, not zero-cost claims.
Reporting adds no fits or objective calls and resets on every tuning call. Default Boxmin and four additional restarts are unchanged.
