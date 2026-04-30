import abc
import numpy as np
from typing import Optional

from ..problem import ProblemABC as Problem

class Sampler(metaclass=abc.ABCMeta):
    """
    Base class for general DOE samplers.

    Subclasses generate samples in the unit hypercube, and the base class
    handles random-state initialization, basic validation, and mapping to the
    problem space.
    """

    def __init__(self):
        self.rng = None

    def sample(self, problem: Problem, nt: int, seed: Optional[int] = None):
        """
        Generate ``nt`` samples and map them from unit space to ``problem`` space.

        :param problem: Problem-like object providing ``nInput`` and ``unit_to_space``.
        :param nt: Number of sample points.
        :param seed: Optional random seed for reproducible sampling.
        :return: A 2D array of shape ``(nt, problem.nInput)`` in the problem space.
        """
        self._validate_problem(problem)
        self._validate_sample_count(nt)

        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        nx = problem.nInput
        X = self._generate(nt, nx)
        X = self._validate_generated_samples(X, nt, nx)

        return problem.unit_to_space(X)

    @abc.abstractmethod
    def _generate(self, nt: int, nx: int):
        """
        Generate unit-hypercube samples with shape (nt, nx).
        """

    def _validate_problem(self, problem: Problem):
        if not isinstance(problem, Problem):
            raise TypeError("problem must be an instance of ProblemABC.")

    def _validate_sample_count(self, nt: int):
        if not isinstance(nt, int):
            raise TypeError("nt must be an integer.")

        if nt <= 0:
            raise ValueError("nt must be greater than 0.")

    def _validate_generated_samples(self, X, nt: int, nx: int):
        """
        Validate the generated unit-space samples before problem-space mapping.
        """
        X = np.asarray(X)

        if X.shape != (nt, nx):
            raise ValueError(f"The generated sample shape must be ({nt}, {nx}).")

        return X

