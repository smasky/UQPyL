#cython: language_level=3
# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause
import numpy as np
cimport numpy as np
cimport cython
import warnings

from scipy.linalg.cython_blas cimport ddot, dasum, daxpy, dnrm2, dcopy, dscal
from scipy.linalg.cython_blas cimport sdot, sasum, saxpy, snrm2, scopy, sscal
from scipy.linalg.cython_lapack cimport sposv, dposv

from numpy.math cimport INFINITY
from cython cimport floating
from libc.math cimport fabs, sqrt, exp, INFINITY, log


cdef:
    int LASSO = 0
    int LOGREG = 1
    int GRPLASSO = 2
    int inc = 1


cdef floating fdot(int * n, floating * x, int * inc1, floating * y,
                        int * inc2) nogil:
    if floating is double:
        return ddot(n, x, inc1, y, inc2)
    else:
        return sdot(n, x, inc1, y, inc2)


cdef floating fasum(int * n, floating * x, int * inc) nogil:
    if floating is double:
        return dasum(n, x, inc)
    else:
        return sasum(n, x, inc)


cdef void faxpy(int * n, floating * alpha, floating * x, int * incx,
                     floating * y, int * incy) nogil:
    if floating is double:
        daxpy(n, alpha, x, incx, y, incy)
    else:
        saxpy(n, alpha, x, incx, y, incy)


cdef floating fnrm2(int * n, floating * x, int * inc) nogil:
    if floating is double:
        return dnrm2(n, x, inc)
    else:
        return snrm2(n, x, inc)


cdef void fcopy(int * n, floating * x, int * incx, floating * y,
                     int * incy) nogil:
    if floating is double:
        dcopy(n, x, incx, y, incy)
    else:
        scopy(n, x, incx, y, incy)


cdef void fscal(int * n, floating * alpha, floating * x,
                     int * incx) nogil:
    if floating is double:
        dscal(n, alpha, x, incx)
    else:
        sscal(n, alpha, x, incx)


cdef void fposv(char * uplo, int * n, int * nrhs, floating * a,
                     int * lda, floating * b, int * ldb, int * info) nogil:
    if floating is double:
        dposv(uplo, n, nrhs, a, lda, b, ldb, info)
    else:
        sposv(uplo, n, nrhs, a, lda, b, ldb, info)


cdef inline floating ST(floating x, floating u) nogil:
    if x > u:
        return x - u
    elif x < - u:
        return x + u
    else:
        return 0


cdef floating log_1pexp(floating x) nogil:
    """Compute log(1. + exp(x)) while avoiding over/underflow."""
    if x < - 18:
        return exp(x)
    elif x > 18:
        return x
    else:
        return log(1. + exp(x))


cdef inline floating xlogx(floating x) nogil:
    if x < 1e-10:
        return 0.
    else:
        return x * log(x)

cdef inline floating Nh(floating x) nogil:
    """Negative entropy of scalar x."""
    if 0. <= x <= 1.:
        return xlogx(x) + xlogx(1. - x)
    else:
        return INFINITY  # not - INFINITY


@cython.boundscheck(False)
@cython.wraparound(False)
cdef floating fweighted_norm_w2(floating[:] w, floating[:] weights) nogil:
    cdef floating weighted_norm = 0.
    cdef int n_features = w.shape[0]
    cdef int j

    for j in range(n_features):
        if weights[j] == INFINITY:
            continue
        weighted_norm += weights[j] * w[j] ** 2
    return weighted_norm


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline floating sigmoid(floating x) nogil:
    return 1. / (1. + exp(- x))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating primal_logreg(
    floating alpha, floating[:] Xw, floating[:] y, floating[:] w,
    floating[:] weights) nogil:
    cdef int inc = 1
    cdef int n_samples = Xw.shape[0]
    cdef int n_features = w.shape[0]
    cdef floating p_obj = 0.
    cdef int i, j
    for i in range(n_samples):
        p_obj += log_1pexp(- y[i] * Xw[i])
    for j in range(n_features):
        # avoid nan when weights[j] is INFINITY
        if w[j]:
            p_obj += alpha * weights[j] * fabs(w[j])
    return p_obj


# todo check normalization by 1 / n_samples everywhere
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating primal_lasso(
        floating alpha, floating l1_ratio, floating[:] R, floating[:] w,
        floating[:] weights) nogil:
    cdef int n_samples = R.shape[0]
    cdef int n_features = w.shape[0]
    cdef int inc = 1
    cdef int j
    cdef floating p_obj = 0.
    p_obj = fdot(&n_samples, &R[0], &inc, &R[0], &inc) / (2. * n_samples)
    for j in range(n_features):
        # avoid nan when weights[j] is INFINITY
        if w[j]:
            p_obj += alpha * weights[j] * (
                     l1_ratio * fabs(w[j]) +
                     0.5 * (1. - l1_ratio) * w[j] ** 2)
    return p_obj


cdef floating primal(
    int pb, floating alpha, floating l1_ratio, floating[:] R, floating[:] y,
    floating[:] w, floating[:] weights) nogil:
    if pb == LASSO:
        return primal_lasso(alpha, l1_ratio, R, w, weights)
    else:
        return primal_logreg(alpha, R, y, w, weights)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating dual_enet(int n_samples, floating alpha, floating l1_ratio,
                         floating norm_y2, floating norm_w2, floating * theta,
                         floating * y) nogil:
    """Theta must be feasible"""
    cdef int i
    cdef floating d_obj = 0.

    for i in range(n_samples):
        d_obj -= (y[i] - n_samples * theta[i]) ** 2
    d_obj *= 0.5 / n_samples
    d_obj += norm_y2 / (2. * n_samples)
    if l1_ratio != 1.0:
        d_obj -= 0.5 * alpha * (1 - l1_ratio) * norm_w2
    return d_obj


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef floating dual_logreg(int n_samples, floating * theta,
                          floating * y) nogil:
    """Compute dual objective value at theta, which must be feasible."""
    cdef int i
    cdef floating d_obj = 0.

    for i in range(n_samples):
        d_obj -= Nh(y[i] * theta[i])
    return d_obj


cdef floating dual(int pb, int n_samples, floating alpha, floating l1_ratio,
                   floating norm_y2, floating norm_w2, floating * theta, floating * y) nogil:
    if pb == LASSO:
        return dual_enet(n_samples, alpha, l1_ratio, norm_y2, norm_w2, &theta[0], &y[0])
    else:
        return dual_logreg(n_samples, &theta[0], &y[0])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void create_dual_pt(
        int pb, int n_samples, floating * out,
        floating * R, floating * y) nogil:
    cdef floating tmp = 1.
    if pb == LASSO:  # out = R / n_samples
        tmp = 1. / n_samples
        fcopy(&n_samples, &R[0], &inc, &out[0], &inc)
    else:  # out = y * sigmoid(-y * Xw)
        for i in range(n_samples):
            out[i] = y[i] * sigmoid(-y[i] * R[i])

    fscal(&n_samples, &tmp, &out[0], &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int create_accel_pt(
    int pb, int n_samples, int epoch, int gap_freq,
    floating * R, floating * out, floating * last_K_R, floating[:, :] U,
    floating[:, :] UtU, floating[:] onesK, floating[:] y):

    # solving linear system in cython
    # doc at https://software.intel.com/en-us/node/468894

    # cdef int n_samples = y.shape[0] cannot use this for MTL
    cdef int K = U.shape[0] + 1
    cdef char * char_U = 'U'
    cdef int one = 1
    cdef int Kminus1 = K - 1
    cdef int inc = 1
    cdef floating sum_z
    cdef int info_dposv

    cdef int i, j, k
    # warning: this is wrong (n_samples) for MTL, it is handled outside
    cdef floating tmp = 1. if pb == LOGREG else 1. / n_samples

    if epoch // gap_freq < K:
        # last_K_R[it // f_gap] = R:
        fcopy(&n_samples, R, &inc,
              &last_K_R[(epoch // gap_freq) * n_samples], &inc)
    else:
        for k in range(K - 1):
            fcopy(&n_samples, &last_K_R[(k + 1) * n_samples], &inc,
                  &last_K_R[k * n_samples], &inc)
        fcopy(&n_samples, R, &inc, &last_K_R[(K - 1) * n_samples], &inc)
        for k in range(K - 1):
            for i in range(n_samples):
                U[k, i] = last_K_R[(k + 1) * n_samples + i] - \
                          last_K_R[k * n_samples + i]

        for k in range(K - 1):
            for j in range(k, K - 1):
                UtU[k, j] = fdot(&n_samples, &U[k, 0], &inc, &U[j, 0], &inc)
                UtU[j, k] = UtU[k, j]

        # refill onesK with ones because it has been overwritten
        # by dposv
        for k in range(K - 1):
            onesK[k] = 1.

        fposv(char_U, &Kminus1, &one, &UtU[0, 0], &Kminus1,
               &onesK[0], &Kminus1, &info_dposv)

        # onesK now holds the solution in x to UtU dot x = onesK
        if info_dposv != 0:
            # don't use accel for this iteration
            for k in range(K - 2):
                onesK[k] = 0
            onesK[K - 2] = 1

        sum_z = 0.
        for k in range(K - 1):
            sum_z += onesK[k]
        for k in range(K - 1):
            onesK[k] /= sum_z

        for i in range(n_samples):
            out[i] = 0.
        for k in range(K - 1):
            for i in range(n_samples):
                out[i] += onesK[k] * last_K_R[k * n_samples + i]

        if pb == LOGREG:
            for i in range(n_samples):
                out[i] = y[i] * sigmoid(- y[i] * out[i])

        fscal(&n_samples, &tmp, &out[0], &inc)
        # out now holds the extrapolated dual point:
        # LASSO: (y - Xw) / n_samples
        # LOGREG:  y * sigmoid(-y * Xw)

    return info_dposv


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void compute_norms_X_col(
        bint is_sparse, floating[:] norms_X_col, int n_samples,
        floating[::1, :] X, floating[:] X_data, int[:] X_indices,
        int[:] X_indptr, floating[:] X_mean):
    cdef int j, startptr, endptr
    cdef floating tmp, X_mean_j
    cdef int n_features = norms_X_col.shape[0]

    for j in range(n_features):
        if is_sparse:
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]
            X_mean_j = X_mean[j]
            tmp = 0.
            for i in range(startptr, endptr):
                tmp += (X_data[i] - X_mean_j) ** 2
            tmp += (n_samples - endptr + startptr) * X_mean_j ** 2
            norms_X_col[j] = sqrt(tmp)
        else:
            norms_X_col[j] = fnrm2(&n_samples, &X[0, j], &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void compute_Xw(
        bint is_sparse, int pb, floating[:] R, floating[:] w,
        floating[:] y, bint center, floating[::1, :] X, floating[:] X_data,
        int[:] X_indices, int[:] X_indptr, floating[:] X_mean):
    # R holds residuals if LASSO, Xw for LOGREG
    cdef int i, j, startptr, endptr
    cdef floating tmp, X_mean_j
    cdef int inc = 1
    cdef int n_samples = y.shape[0]
    cdef int n_features = w.shape[0]

    for j in range(n_features):
        if w[j] != 0:
            if is_sparse:
                startptr, endptr = X_indptr[j], X_indptr[j + 1]
                for i in range(startptr, endptr):
                    R[X_indices[i]] += w[j] * X_data[i]
                if center:
                    X_mean_j = X_mean[j]
                    for i in range(n_samples):
                        R[i] -= X_mean_j * w[j]
            else:
                tmp = w[j]
                faxpy(&n_samples, &tmp, &X[0, j], &inc, &R[0], &inc)
    # currently R = X @ w, update for LASSO/GRPLASSO:
    if pb in (LASSO, GRPLASSO):
        for i in range(n_samples):
            R[i] = y[i] - R[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef floating dnorm_enet(
        bint is_sparse, floating[:] theta, floating[:] w, floating[::1, :] X,
        floating[:] X_data, int[:] X_indices, int[:] X_indptr, int[:] skip,
        floating[:] X_mean, floating[:] weights, bint center,
        bint positive, floating alpha, floating l1_ratio) nogil:
    """compute norm(X[:, ~skip].T.dot(theta), ord=inf)"""
    cdef int n_samples = theta.shape[0]
    cdef int n_features = skip.shape[0]
    cdef floating Xj_theta
    cdef floating dnorm_XTtheta = 0.
    cdef floating theta_sum = 0.
    cdef int i, j, Cj, startptr, endptr

    if is_sparse:
        # TODO by design theta_sum should always be 0 when center
        if center:
            for i in range(n_samples):
                theta_sum += theta[i]

    # max over feature for which skip[j] == False
    for j in range(n_features):
        if skip[j] or weights[j] == INFINITY:
            continue
        if is_sparse:
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]
            Xj_theta = 0.
            for i in range(startptr, endptr):
                Xj_theta += X_data[i] * theta[X_indices[i]]
            if center:
                Xj_theta -= theta_sum * X_mean[j]
        else:
            Xj_theta = fdot(&n_samples, &theta[0], &inc, &X[0, j], &inc)

        # minus sign to consider the choice theta = y - Xw and not theta = Xw -y
        if l1_ratio != 1:
            Xj_theta -= alpha * (1 - l1_ratio) * weights[j] * w[j]

        if not positive:
            Xj_theta = fabs(Xj_theta)
        dnorm_XTtheta = max(dnorm_XTtheta, Xj_theta / weights[j])
    return dnorm_XTtheta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void set_prios(
    bint is_sparse, floating[:] theta, floating[:] w, floating alpha, floating l1_ratio,
    floating[::1, :] X, floating[:] X_data, int[:] X_indices, int[:] X_indptr,
    floating[:] norms_X_col, floating[:] weights, floating[:] prios,
    int[:] screened, floating radius, int * n_screened, bint positive) nogil:
    cdef int i, j, startptr, endptr
    cdef floating Xj_theta
    cdef int n_samples = theta.shape[0]
    cdef int n_features = prios.shape[0]
    cdef floating norms_X_col_j = 0.

    # TODO we do not substract theta_sum, which seems to indicate that theta
    # is always centered...
    for j in range(n_features):
        if screened[j] or norms_X_col[j] == 0. or weights[j] == 0.:
            prios[j] = INFINITY
            continue
        if is_sparse:
            Xj_theta = 0
            startptr = X_indptr[j]
            endptr = X_indptr[j + 1]
            for i in range(startptr, endptr):
                Xj_theta += theta[X_indices[i]] * X_data[i]
        else:
            Xj_theta = fdot(&n_samples, &theta[0], &inc, &X[0, j], &inc)

        norms_X_col_j = norms_X_col[j]
        if l1_ratio != 1:
            Xj_theta -= alpha * (1 - l1_ratio) * weights[j] * w[j]

            norms_X_col_j = norms_X_col_j ** 2
            norms_X_col_j += sqrt(norms_X_col_j + alpha * (1 - l1_ratio) * weights[j])

        if positive:
            prios[j] = fabs(Xj_theta - alpha * l1_ratio * weights[j]) / norms_X_col_j
        else:
            prios[j] = (alpha * l1_ratio * weights[j] - fabs(Xj_theta)) / norms_X_col_j

        if prios[j] > radius:
            screened[j] = True
            n_screened[0] += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def celer(
        bint is_sparse, int pb, floating[::1, :] X, floating[:] X_data,
        int[:] X_indices, int[:] X_indptr, floating[:] X_mean,
        floating[:] y, floating alpha, floating l1_ratio, floating[:] w, floating[:] Xw,
        floating[:] theta, floating[:] norms_X_col, floating[:] weights,
        int max_iter, int max_epochs, int gap_freq=10,
        float tol=1e-6, int p0=100, int verbose=0,
        int use_accel=1, int prune=0, bint positive=0,
        int better_lc=1):
    """R/Xw and w are modified in place and assumed to match.
    Weights must be > 0, features with weights equal to np.inf are ignored.
    WARNING for Logreg the datafit is a sum, while for Lasso it is a mean.
    """
    assert pb in (LASSO, LOGREG)

    if floating is double:
        dtype = np.float64
    else:
        dtype = np.float32

    cdef int inc = 1
    cdef int verbose_in = max(0, verbose - 1)
    cdef int n_features = w.shape[0]
    cdef int n_samples = y.shape[0]

    # scale stopping criterion: multiply tol by primal value at w = 0
    if pb == LASSO:
        # actually for Lasso, omit division by 2 to match sklearn
        tol *= fnrm2(&n_samples, &y[0], &inc) ** 2 / n_samples
    elif pb == LOGREG:
        tol *= n_samples * np.log(2)

    if p0 > n_features:
        p0 = n_features

    cdef int t = 0
    cdef int i, j, k, idx, startptr, endptr, epoch
    cdef int ws_size = 0
    cdef int nnz = 0
    cdef floating gap = -1  # initialized for the warning if max_iter=0
    cdef floating p_obj, d_obj, highest_d_obj, radius, tol_in
    cdef floating gap_in, p_obj_in, d_obj_in, d_obj_accel, highest_d_obj_in
    cdef floating theta_scaling, R_sum, tmp, tmp_exp, dnorm_XTtheta
    cdef int n_screened = 0
    cdef bint center = False
    cdef floating old_w_j, X_mean_j
    cdef floating[:] prios = np.empty(n_features, dtype=dtype)
    cdef int[:] screened = np.zeros(n_features, dtype=np.int32)
    cdef int[:] notin_ws = np.zeros(n_features, dtype=np.int32)


    # acceleration variables:
    cdef int K = 6
    cdef floating[:, :] last_K_Xw = np.empty([K, n_samples], dtype=dtype)
    cdef floating[:, :] U = np.empty([K - 1, n_samples], dtype=dtype)
    cdef floating[:, :] UtU = np.empty([K - 1, K - 1], dtype=dtype)
    cdef floating[:] onesK = np.ones(K - 1, dtype=dtype)
    cdef int info_dposv

    if is_sparse:
        # center = X_mean.any():
        for j in range(n_features):
            if X_mean[j]:
                center = True
                break

    # TODO this is used only for logreg, L97 is misleading and deserves a comment/refactoring
    cdef floating[:] inv_lc = np.zeros(n_features, dtype=dtype)

    for j in range(n_features):
        # can have 0 features when performing CV on sparse X
        if norms_X_col[j]:
            if pb == LOGREG:
                inv_lc[j] = 4. / norms_X_col[j] ** 2
            else:
                inv_lc[j] = 1. / norms_X_col[j] ** 2

    cdef floating norm_y2 = fnrm2(&n_samples, &y[0], &inc) ** 2
    cdef floating weighted_norm_w2 = fweighted_norm_w2(w, weights)
    theta_scaling = 1.0

    # max_iter + 1 is to deal with max_iter=0
    cdef floating[:] gaps = np.zeros(max_iter + 1, dtype=dtype)
    gaps[0] = -1

    cdef floating[:] theta_in = np.zeros(n_samples, dtype=dtype)
    cdef floating[:] thetacc = np.zeros(n_samples, dtype=dtype)
    cdef floating d_obj_from_inner = 0.

    cdef int[:] ws
    cdef int[:] all_features = np.arange(n_features, dtype=np.int32)

    for t in range(max_iter):
        if t != 0:
            create_dual_pt(pb, n_samples, &theta[0], &Xw[0], &y[0])

            dnorm_XTtheta = dnorm_enet(
                is_sparse, theta, w, X, X_data, X_indices, X_indptr, screened,
                X_mean, weights, center, positive, alpha, l1_ratio)

            if dnorm_XTtheta > alpha * l1_ratio:
                theta_scaling = alpha * l1_ratio / dnorm_XTtheta
                fscal(&n_samples, &theta_scaling, &theta[0], &inc)
            else:
                theta_scaling = 1.

            #  compute ||w||^2 only for Enet
            if l1_ratio != 1:
                weighted_norm_w2 = fweighted_norm_w2(w, weights)

            d_obj = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                theta_scaling**2*weighted_norm_w2, &theta[0], &y[0])

            # also test dual point returned by inner solver after 1st iter:
            dnorm_XTtheta = dnorm_enet(
                is_sparse, theta_in, w, X, X_data, X_indices, X_indptr,
                screened, X_mean, weights, center, positive, alpha, l1_ratio)

            if dnorm_XTtheta  > alpha * l1_ratio:
                theta_scaling = alpha * l1_ratio / dnorm_XTtheta
                fscal(&n_samples, &theta_scaling, &theta_in[0], &inc)
            else:
                theta_scaling = 1.

            d_obj_from_inner = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                    theta_scaling**2*weighted_norm_w2, &theta_in[0], &y[0])
        else:
            d_obj = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                theta_scaling**2*weighted_norm_w2, &theta[0], &y[0])

        if d_obj_from_inner > d_obj:
            d_obj = d_obj_from_inner
            fcopy(&n_samples, &theta_in[0], &inc, &theta[0], &inc)

        highest_d_obj = d_obj  # TODO monotonicity could be enforced but it
        # would add yet another variable, best_theta. I'm not sure it brings
        # anything.

        p_obj = primal(pb, alpha, l1_ratio, Xw, y, w, weights)
        gap = p_obj - highest_d_obj
        gaps[t] = gap
        if verbose:
            print("Iter %d: primal %.10f, gap %.2e" % (t, p_obj, gap), end="")

        if gap <= tol:
            if verbose:
                print("\nEarly exit, gap: %.2e < %.2e" % (gap, tol))
            break

        if pb == LASSO:
            radius = sqrt(2 * gap / n_samples)
        else:
            radius = sqrt(gap / 2.)
        set_prios(
            is_sparse, theta, w, alpha, l1_ratio, X, X_data, X_indices, X_indptr, norms_X_col,
            weights, prios, screened, radius, &n_screened, positive)

        if prune:
            nnz = 0
            for j in range(n_features):
                if w[j] != 0:
                    prios[j] = -1.
                    nnz += 1

            if t == 0:
                ws_size = p0 if nnz == 0 else nnz
            else:
                ws_size = 2 * nnz

        else:
            for j in range(n_features):
                if w[j] != 0:
                    prios[j] = - 1  # include active features
            if t == 0:
                ws_size = p0
            else:
                for j in range(ws_size):
                    if not screened[ws[j]]:
                        # include previous features, if not screened
                        prios[ws[j]] = -1
                ws_size = 2 * ws_size
        if ws_size > n_features - n_screened:
            ws_size = n_features - n_screened


        # if ws_size === n_features then argpartition will break:
        if ws_size == n_features:
            ws = all_features
        else:
            ws = np.argpartition(np.asarray(prios), ws_size)[:ws_size].astype(np.int32)

        for j in range(n_features):
            notin_ws[j] = 1
        for idx in range(ws_size):
            notin_ws[ws[idx]] = 0

        if prune:
            tol_in = 0.3 * gap
        else:
            tol_in = tol

        if verbose:
            print(", %d feats in subpb (%d left)" %
                  (len(ws), n_features - n_screened))

        # calling inner solver which will modify w and R inplace
        highest_d_obj_in = 0
        for epoch in range(max_epochs):
            if epoch != 0 and epoch % gap_freq == 0:
                create_dual_pt(
                    pb, n_samples, &theta_in[0], &Xw[0], &y[0])

                dnorm_XTtheta  = dnorm_enet(
                    is_sparse, theta_in, w, X, X_data, X_indices, X_indptr,
                    notin_ws, X_mean, weights, center, positive, alpha, l1_ratio)

                if dnorm_XTtheta  > alpha * l1_ratio:
                    theta_scaling = alpha * l1_ratio / dnorm_XTtheta
                    fscal(&n_samples, &theta_scaling, &theta_in[0], &inc)
                else:
                    theta_scaling = 1.

                # update norm_w2 in inner loop for Enet only
                if l1_ratio != 1:
                    weighted_norm_w2 = fweighted_norm_w2(w, weights)
                d_obj_in = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                    theta_scaling**2*weighted_norm_w2, &theta_in[0], &y[0])

                if use_accel: # also compute accelerated dual_point
                    info_dposv = create_accel_pt(
                        pb, n_samples, epoch, gap_freq, &Xw[0],
                        &thetacc[0], &last_K_Xw[0, 0], U, UtU, onesK, y)

                    if info_dposv != 0 and verbose_in:
                        pass
                        # print("linear system solving failed")

                    if epoch // gap_freq >= K:
                        dnorm_XTtheta  = dnorm_enet(
                            is_sparse, thetacc, w, X, X_data, X_indices,
                            X_indptr, notin_ws, X_mean, weights, center,
                            positive, alpha, l1_ratio)

                        if dnorm_XTtheta  > alpha * l1_ratio:
                            theta_scaling = alpha * l1_ratio / dnorm_XTtheta
                            fscal(&n_samples, &theta_scaling, &thetacc[0], &inc)
                        else:
                            theta_scaling = 1.

                        d_obj_accel = dual(pb, n_samples, alpha, l1_ratio, norm_y2,
                                theta_scaling**2*weighted_norm_w2, &thetacc[0], &y[0])
                        if d_obj_accel > d_obj_in:
                            d_obj_in = d_obj_accel
                            fcopy(&n_samples, &thetacc[0], &inc,
                                  &theta_in[0], &inc)

                if d_obj_in > highest_d_obj_in:
                    highest_d_obj_in = d_obj_in

                # CAUTION: code does not yet include a best_theta.
                # Can be an issue in screening: dgap and theta might disagree.

                p_obj_in = primal(pb, alpha, l1_ratio, Xw, y, w, weights)
                gap_in = p_obj_in - highest_d_obj_in

                if verbose_in:
                    print("Epoch %d, primal %.10f, gap: %.2e" %
                          (epoch, p_obj_in, gap_in))
                if gap_in < tol_in:
                    if verbose_in:
                        print("Exit epoch %d, gap: %.2e < %.2e" % \
                              (epoch, gap_in, tol_in))
                    break

            for k in range(ws_size):
                j = ws[k]
                if norms_X_col[j] == 0. or weights[j] == INFINITY:
                    continue
                old_w_j = w[j]

                if pb == LASSO:
                    if is_sparse:
                        X_mean_j = X_mean[j]
                        startptr, endptr = X_indptr[j], X_indptr[j + 1]
                        for i in range(startptr, endptr):
                            w[j] += Xw[X_indices[i]] * X_data[i] / \
                                    norms_X_col[j] ** 2
                        if center:
                            R_sum = 0.
                            for i in range(n_samples):
                                R_sum += Xw[i]
                            w[j] -= R_sum * X_mean_j / norms_X_col[j] ** 2
                    else:
                        w[j] += fdot(&n_samples, &X[0, j], &inc, &Xw[0],
                                     &inc) / norms_X_col[j] ** 2

                    if positive and w[j] <= 0.:
                        w[j] = 0.
                    else:
                        if l1_ratio != 1.:
                            w[j] = ST(
                                w[j],
                                alpha * l1_ratio / norms_X_col[j] ** 2 * n_samples * weights[j]) / \
                                (1 + alpha * (1 - l1_ratio) * weights[j] /  norms_X_col[j] ** 2 * n_samples)
                        else:
                            w[j] = ST(
                                w[j],
                                alpha / norms_X_col[j] ** 2 * n_samples * weights[j])

                    # R -= (w_j - old_w_j) * (X[:, j] - X_mean[j])
                    tmp = old_w_j - w[j]
                    if tmp != 0.:
                        if is_sparse:
                            for i in range(startptr, endptr):
                                Xw[X_indices[i]] += tmp * X_data[i]
                            if center:
                                for i in range(n_samples):
                                    Xw[i] -= X_mean_j * tmp
                        else:
                            faxpy(&n_samples, &tmp, &X[0, j], &inc,
                                  &Xw[0], &inc)
                else:
                    if is_sparse:
                        startptr = X_indptr[j]
                        endptr = X_indptr[j + 1]
                        if better_lc:
                            tmp = 0.
                            for i in range(startptr, endptr):
                                tmp_exp = exp(Xw[X_indices[i]])
                                tmp += X_data[i] ** 2 * tmp_exp / \
                                       (1. + tmp_exp) ** 2
                            inv_lc[j] = 1. / tmp
                    else:
                        if better_lc:
                            tmp = 0.
                            for i in range(n_samples):
                                tmp_exp = exp(Xw[i])
                                tmp += (X[i, j] ** 2) * tmp_exp / \
                                       (1. + tmp_exp) ** 2
                            inv_lc[j] = 1. / tmp

                    tmp = 0.  # tmp = dot(Xj, y * sigmoid(-y * w)) / lc[j]
                    if is_sparse:
                        for i in range(startptr, endptr):
                            idx = X_indices[i]
                            tmp += X_data[i] * y[idx] * \
                                   sigmoid(- y[idx] * Xw[idx])
                    else:
                        for i in range(n_samples):
                            tmp += X[i, j] * y[i] * sigmoid(- y[i] * Xw[i])

                    w[j] = ST(w[j] + tmp * inv_lc[j],
                              alpha * inv_lc[j] * weights[j])

                    tmp = w[j] - old_w_j
                    if tmp != 0.:
                        if is_sparse:
                            for i in range(startptr, endptr):
                                Xw[X_indices[i]] += tmp * X_data[i]
                        else:
                            faxpy(&n_samples, &tmp, &X[0, j], &inc,
                                  &Xw[0], &inc)
        else:
            warnings.warn(
                'Inner solver did not converge at ' +
                f'epoch: {epoch}, gap: {gap_in:.2e} > {tol_in:.2e}')
    else:
        warnings.warn(
            'Objective did not converge: duality ' +
            f'gap: {gap}, tolerance: {tol}. Increasing `tol` may make the' +
            ' solver faster without affecting the results much. \n' +
            'Fitting data with very small alpha causes precision issues.')

    return np.asarray(w), np.asarray(theta), np.asarray(gaps[:t + 1])

