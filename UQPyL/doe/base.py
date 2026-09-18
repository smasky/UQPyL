import abc
import numpy as np
from typing import Optional

from ..problem import ProblemABC as Problem

class Sampler(metaclass=abc.ABCMeta):
    """
    Base class for DOE samplers.
    """

    def __init__(self):
        self.rng = None

    def sample(self, problem: Problem, nSamples: Optional[int] = None, seed: Optional[int] = None, nt: Optional[int] = None, *, output="real"):
        """
        Generate samples in the problem space.

        :param problem: Problem instance.
        :param nSamples: Number of samples.
        :param seed: Random seed.
        :param output: "real" for decoded samples (default), or "unit" for unit coordinates.
        :return np.ndarray: Samples in the selected coordinate space.
        """
        nSamples = self._resolve_sample_count(nSamples=nSamples, nt=nt)
        X, _ = self.sampleWithMeta(problem, nSamples, seed=seed, output=output)
        return X

    def sampleWithMeta(self, problem: Problem, nSamples: Optional[int] = None, seed: Optional[int] = None, nt: Optional[int] = None, *, output="real"):
        """
        Generate samples with metadata.

        :param problem: Problem instance.
        :param nSamples: Number of samples.
        :param seed: Random seed.
        :param output: "real" for decoded samples (default), or "unit" for unit coordinates.
        :return tuple: ``(X, meta)`` with the output space recorded in metadata.
        """
        nSamples = self._resolve_sample_count(nSamples=nSamples, nt=nt)
        self._validate_problem(problem)
        self._validate_sample_count(nSamples)

        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        nInput = problem.nInput
        U = self._generate(nSamples, nInput)
        expected_shape = self._expected_shape(nSamples, nInput)
        U = self._validate_generated_samples(U, expected_shape)
        X = self._select_output(problem, U, output)

        meta = self._build_meta(problem, nSamples, seed=seed)
        meta["output"] = output
        return X, meta

    @staticmethod
    def _select_output(problem, U, output):
        if output == "unit":
            return U.copy()
        if output == "real":
            return problem.unit_to_space(U)
        raise ValueError("output must be 'real' or 'unit'.")

    def _resolve_sample_count(self, nSamples: Optional[int] = None, nt: Optional[int] = None):
        if nSamples is None:
            nSamples = nt
        elif nt is not None and nt != nSamples:
            raise ValueError("nSamples and nt must match when both are provided.")

        if nSamples is None:
            raise TypeError("nSamples (or legacy alias nt) must be provided.")

        return nSamples

    @abc.abstractmethod
    def _generate(self, nSamples: int, nInput: int):
        """
        Generate unit-space samples.

        :param nSamples: Number of samples.
        :param nInput: Number of input variables.
        :return np.ndarray: Unit-space samples.
        """

    def _build_meta(self, problem: Problem, nSamples: int, seed: Optional[int] = None):
        """
        Build sample metadata.

        :param problem: Problem instance.
        :param nSamples: Number of samples.
        :param seed: Random seed.
        :return dict: Sampling metadata.
        """
        return {}

    def _expected_shape(self, nSamples: int, nInput: int):
        """
        Return the expected unit-space shape.

        :param nSamples: Number of samples.
        :param nInput: Number of input variables.
        :return tuple: Expected sample shape.
        """
        return (nSamples, nInput)

    def _validate_problem(self, problem: Problem):
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of ProblemABC.")

    def _validate_sample_count(self, nSamples: int):
        if not isinstance(nSamples, int):
            raise TypeError("nSamples must be an integer.")

        if nSamples <= 0:
            raise ValueError("nSamples must be greater than 0.")

    def _validate_generated_samples(self, X, expected_shape):
        """
        Validate the generated unit-space samples.

        :param X: Generated samples.
        :param expected_shape: Expected array shape.
        :return np.ndarray: Validated samples.
        """
        X = np.asarray(X)

        if X.shape != expected_shape:
            raise ValueError(f"The generated sample shape must be {expected_shape}.")

        return X

