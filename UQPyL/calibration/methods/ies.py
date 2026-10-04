import numpy as np
import warnings

from ._ensemble import validateRegularization, anomalyGain, regressionSensitivity
from .es import ES


class IES(ES):
    """Prior-anchored, stochastic Gauss-Newton ensemble smoother (EnRML).

    The original ensemble and perturbed observations stay fixed throughout a
    run. ``lam`` is dimensionless damping of the prior-metric Gauss-Newton
    Hessian, not extra observation noise. Zero selects the full GN step.
    This is fixed damping, not an adaptive LM acceptance/rejection algorithm.
    Optional ``adaptive=True`` backtracks the step against the fixed RML
    objective and stops on small steps or a failed line search.

    References:
        Raanes, Stordal and Evensen (2019), Revising the stochastic iterative
        ensemble smoother, Nonlin. Processes Geophys., 26, 325-338.

    Notes:
        Finite ensembles approximate the posterior. Box clipping changes the
        unconstrained update. Zero/singular R uses a pseudoinverse extension;
        lost regression directions retain their last estimated slope. This
        extension does not guarantee satisfaction of inconsistent hard data.
        The scoring metric only controls reporting and best-member selection.
    """

    name = "IES"

    def __init__(
        self,
        verboseFlag: bool = False,
        verboseFreq: int = 1,
        saveFlag: bool = False,
        logFlag: bool = False,
        maxIters: int = 5,
        lam: float = 0.0,
        metric="rmse",
        seed: int | None = None,
        adaptive: bool = False,
        tolerance: float = 1e-6,
        maxBacktracks: int = 8,
        boundHandling: str = "clip",
        localLinearization: bool = False,
    ):
        super().__init__(
            verboseFlag=verboseFlag,
            verboseFreq=verboseFreq,
            saveFlag=saveFlag,
            logFlag=logFlag,
            metric=metric,
            boundHandling=boundHandling,
        )
        self.set("maxIters", maxIters)
        self.set("lam", validateRegularization(lam))
        self.set("seed", seed)
        self.set("adaptive", adaptive)
        self.set("tolerance", tolerance)
        self.set("maxBacktracks", maxBacktracks)
        self.set("localLinearization", localLinearization)

    def _prepareIteration(self, X, obs, r):
        """Build immutable prior quantities and draw noise once per run."""
        anomalies = X - X.mean(axis=0)
        scales = np.max(np.abs(anomalies), axis=0)
        scales = np.where(scales > 0, scales, 1.0)
        normalized = anomalies / scales
        targets = np.broadcast_to(obs, (len(X), len(obs))).copy()
        if r is not None and np.any(r):
            values, vectors = np.linalg.eigh(r)
            noiseRoot = vectors * np.sqrt(np.maximum(values, 0))
            rng = np.random.default_rng(self.get("seed"))
            targets += rng.standard_normal(targets.shape) @ noiseRoot.T
        return {
            "prior": X.copy(),
            "anomalies": anomalies,
            "scales": scales,
            "targets": targets,
            "referenceNorm": float(np.linalg.norm(normalized)),
            "sensitivity": np.zeros((X.shape[1], len(obs))),
        }

    def _runCore(self, problem, X, r: np.ndarray | None = None):
        """Smooth an initial ensemble with optional observation covariance R."""
        X_cur = self._validateEnsemble(X).copy()
        maxIters = self.get("maxIters")
        if isinstance(maxIters, (bool, np.bool_)) or not isinstance(maxIters, (int, np.integer)) or maxIters < 0:
            raise ValueError("maxIters must be a nonnegative integer.")
        lam = validateRegularization(self.get("lam"))
        obs, r = self._prepareObservations(r)
        adaptive, tolerance, maxBacktracks = self.get("adaptive"), self.get("tolerance"), self.get("maxBacktracks")
        if not isinstance(adaptive, (bool, np.bool_)):
            raise ValueError("adaptive must be boolean.")
        if not isinstance(self.get("localLinearization"), (bool, np.bool_)):
            raise ValueError("localLinearization must be boolean.")
        if np.ndim(tolerance) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative.")
        if (
            isinstance(maxBacktracks, (bool, np.bool_))
            or not isinstance(maxBacktracks, (int, np.integer))
            or maxBacktracks < 0
        ):
            raise ValueError("maxBacktracks must be a nonnegative integer.")
        context = self._prepareIteration(X_cur, obs, r)
        sim_post_flat = self.evaluate(X_cur, validOnly=False)
        priorScores = self.score(sim_post_flat)
        self.state.diagnostics["updateMethod"] = "prior_anchored_enrml"
        self.state.diagnostics["damping"] = lam
        self.state.diagnostics["seed"] = self.get("seed")
        self.state.diagnostics["linearization"] = (
            "member_finite_difference" if self.get("localLinearization") else "ensemble_regression"
        )
        merit = self._makeMerit(context, r, sim_post_flat) if adaptive else None
        stopReason = "iteration_budget"

        for iterIdx in range(maxIters):
            X_next, scores, meta = self._updatePrepared(X_cur, obs, r, lam, sim_post_flat, context)
            step = 1.0
            if adaptive:
                before = merit(X_cur, sim_post_flat)
                accepted = False
                for backtrack in range(maxBacktracks + 1):
                    if backtrack:
                        step *= 0.5
                        X_next = self._boundUpdate(X_cur + step * (meta["analysis"] - X_cur), X_cur)
                        meta["posteriorSims"] = self.evaluate(X_next, validOnly=False)
                        scores = self.score(meta["posteriorSims"])
                    after = merit(X_next, meta["posteriorSims"])
                    hardTolerance = 1e-12 * max(1.0, before[0])
                    accepted = bool(
                        np.isfinite(after).all()
                        and (
                            after[0] < before[0] - hardTolerance
                            or (
                                after[0] <= before[0] + hardTolerance
                                and after[1] <= before[1] + 1e-12 * max(1.0, before[1])
                            )
                        )
                    )
                    self.state.diagnostics["boundUpdates"][-1]["accepted"] = accepted
                    self.state.diagnostics.setdefault("lineSearch", []).append(
                        {
                            "iteration": iterIdx + 1,
                            "step": step,
                            "merit_before": list(before),
                            "merit_after": list(after),
                            "accepted": accepted,
                        }
                    )
                    if accepted:
                        break
                if not accepted:
                    stopReason = "line_search_stalled"
                    warnings.warn(
                        "IES line search found no improving step; retaining the last accepted ensemble.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    break
            relativeStep = float(np.max(np.abs((X_next - X_cur) / context["scales"])))
            sim_post_flat = meta["posteriorSims"]
            self.state.diagnostics.setdefault("covarianceSolves", []).append(meta["covarianceSolve"])
            self.state.diagnostics.setdefault("regressionRanks", []).append(meta["regressionRank"])
            self.state.history.metricsHistory.append({"iter": iterIdx + 1, "scoreMean": float(np.mean(scores))})
            X_cur = X_next
            if adaptive and relativeStep <= tolerance:
                stopReason = "step_tolerance"
                break

        self.state.diagnostics["stopReason"] = stopReason

        scores = self.score(sim_post_flat)
        bestIdx = int(np.argmin(self.normalizedScore(sim_post_flat)))
        self.recordPosterior(X_cur.copy(), sim_post_flat.copy())
        self.recordBest(X_cur[bestIdx : bestIdx + 1], sim_post_flat[bestIdx : bestIdx + 1])
        self.state.diagnostics["priorMean"] = context["prior"].mean(axis=0)
        self.state.diagnostics["posteriorMean"] = X_cur.mean(axis=0)
        self.state.diagnostics["priorScores"] = priorScores
        self.state.diagnostics["scores"] = scores
        self.state.extra["bestIdx"] = bestIdx

    def _update_once(self, X, r: np.ndarray | None = None, lam: float = 0.0, fullSim=None):
        """Apply one GN step, using X as the prior for this standalone call."""
        X = self._validateEnsemble(X)
        obs, r = self._prepareObservations(r)
        lam = validateRegularization(lam)
        context = self._prepareIteration(X, obs, r)
        if fullSim is None:
            fullSim = self.evaluate(X, validOnly=False)
        return self._updatePrepared(X, obs, r, lam, fullSim, context)

    def _updatePrepared(self, X, obs, r, lam, fullSim, context):
        Y = fullSim[:, self.getValidMask()]
        if not np.all(np.isfinite(Y)):
            raise ValueError("Ensemble simulations must be finite.")
        if self.get("localLinearization"):
            analysis, solveInfo = self._localAnalysis(X, Y, r, lam, context)
            rank = None
        else:
            sensitivity, rank = regressionSensitivity(
                X, Y, context["scales"], context["sensitivity"], context["referenceNorm"]
            )
            context["sensitivity"] = sensitivity
            projected = (context["anomalies"] / context["scales"]) @ sensitivity
            # Inverting (1+lam) C_prior^-1 + H.T R^-1 H in gain form
            # scales R by (1+lam), but does not alter the fixed perturbed targets.
            noise = None if r is None else r * (1.0 + lam)
            gain, solveInfo = anomalyGain(context["anomalies"], projected, noise)
            displacement = (X - context["prior"]) / (1.0 + lam)
            innovation = context["targets"] - Y + (displacement / context["scales"]) @ sensitivity
            analysis = X - displacement + innovation @ gain.T
        X_post = self._boundUpdate(analysis, X)
        sim_post_flat = self.problem.flattenSim(self.problem.simFunc(X_post))
        scores = self.score(sim_post_flat)
        return (
            X_post,
            scores,
            {
                "priorScores": self.score(fullSim),
                "covarianceSolve": solveInfo,
                "posteriorSims": sim_post_flat,
                "regressionRank": rank,
                "analysis": analysis,
            },
        )

    def _localAnalysis(self, X, Y, r, lam, context):
        """Optional memberwise finite-difference RML, costing up to 2*p batches.

        Unlike one ensemble regression slope, each member gets its own local
        derivative. This changes the nonlinear approximation, not the prior or
        target. It is not a claim of exact nonlinear posterior sampling.
        """
        derivatives = np.zeros((len(X), Y.shape[1], X.shape[1]))
        for column in range(X.shape[1]):
            if self.problem.lb[0, column] == self.problem.ub[0, column]:
                continue
            step = np.cbrt(np.finfo(float).eps) * context["scales"][column]
            plus, minus = X.copy(), X.copy()
            plus[:, column] = np.minimum(X[:, column] + step, self.problem.ub[0, column])
            minus[:, column] = np.maximum(X[:, column] - step, self.problem.lb[0, column])
            difference = plus[:, column] - minus[:, column]
            if np.any(difference <= 0):
                raise ValueError("Local finite-difference step is not representable; rescale parameters.")
            derivatives[:, :, column] = (self.evaluate(plus) - self.evaluate(minus)) / difference[:, None]
            self.state.diagnostics["derivativeBatches"] = self.state.diagnostics.get("derivativeBatches", 0) + 2
        noise = None if r is None else r * (1 + lam)
        displacement = (X - context["prior"]) / (1 + lam)
        analysis = np.empty_like(X)
        infos = []
        for index, derivative in enumerate(derivatives):
            gain, info = anomalyGain(context["anomalies"], context["anomalies"] @ derivative.T, noise)
            innovation = context["targets"][index] - Y[index] + derivative @ displacement[index]
            analysis[index] = X[index] - displacement[index] + gain @ innovation
            infos.append(info)
        return analysis, {"solver": "memberwise", "members": infos}

    def _makeMerit(self, context, r, initialSim):
        """Fixed RML objective; hard-data residual takes lexicographic priority.

        The prior term uses the initial ensemble subspace. A clipped proposal
        outside that support is rejected instead of assigning it zero penalty.
        """
        anomalies = context["anomalies"] / context["scales"]
        _, singular, right = np.linalg.svd(anomalies, full_matrices=False)
        cutoff = max(anomalies.shape) * np.finfo(float).eps * singular.max(initial=0)
        active = singular > cutoff
        basis = right[active].T
        priorRoot = singular[active] / np.sqrt(len(anomalies) - 1)
        if r is None:
            softBasis, softRoot, hardBasis = None, None, None
        else:
            values, vectors = np.linalg.eigh(r)
            active = values > len(r) * np.finfo(float).eps * values.max(initial=0)
            softBasis, softRoot, hardBasis = vectors[:, active], np.sqrt(values[active]), vectors[:, ~active]
        initialResponses = initialSim[:, self.getValidMask()]
        dataScale = max(np.max(np.abs(initialResponses)), np.max(np.abs(context["targets"])), np.finfo(float).tiny)

        def merit(X, simulation):
            displacement = (X - context["prior"]) / context["scales"]
            coordinates = displacement @ basis
            if np.max(np.abs(displacement - coordinates @ basis.T), initial=0) > 1e-10 * max(
                1.0, np.max(np.abs(displacement))
            ):
                return np.array([np.inf, np.inf])
            priorCost = np.mean(np.sum((coordinates / priorRoot) ** 2, axis=1))
            residual = simulation[:, self.getValidMask()] - context["targets"]
            if r is None:
                hard, soft = residual / dataScale, 0.0
            else:
                hard = (residual / dataScale) @ hardBasis
                soft = np.mean(np.sum((residual @ softBasis / softRoot) ** 2, axis=1))
            return np.array([np.mean(np.sum(hard**2, axis=1)), priorCost + soft])

        return merit
