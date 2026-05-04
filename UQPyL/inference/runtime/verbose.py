from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VerboseConfig:
    level: int = 2
    progressEvery: int = 1
    summaryEvery: int = 50
    precision: int = 4
    useTTY: bool = True


@dataclass
class ProgressState:
    iteration: int
    nEval: int
    elapsed: float
    meanLogProb: float | None
    currentLogProb: float | None
    acceptanceRateMean: float
    feasibleRate: float
    bestValue: float | None
    nChains: int
    draws: int
    acceptanceRate: np.ndarray | None = None
    bestX: np.ndarray | None = None
    meanX: np.ndarray | None = None
    stdX: np.ndarray | None = None
    params: dict[str, Any] | None = None


class InferenceRenderer:
    def __init__(self, config):
        self.config = config

    def renderProgress(self, state, methodName):
        return (
            f"{methodName} | iter={state.iteration} eval={state.nEval} "
            f"curLogp={_fmt(state.currentLogProb, self.config.precision)} "
            f"accept={_fmt(state.acceptanceRateMean, self.config.precision)} "
            f"feasible={_fmt(state.feasibleRate, self.config.precision)} "
            f"best={_fmt(state.bestValue, self.config.precision)} "
            f"time={state.elapsed:.1f}s"
        )

    def renderSummary(self, state, methodName, fullVectors=False):
        pairs = [
            ("iter", state.iteration),
            ("evaluations", state.nEval),
            ("chains", state.nChains),
            ("draws", state.draws),
            ("current logProb", _fmt(state.currentLogProb, self.config.precision)),
            ("mean logProb", _fmt(state.meanLogProb, self.config.precision)),
            ("acceptance mean", _fmt(state.acceptanceRateMean, self.config.precision)),
            ("acceptance min", _fmt(_nan_min(state.acceptanceRate), self.config.precision)),
            ("acceptance max", _fmt(_nan_max(state.acceptanceRate), self.config.precision)),
            ("feasible rate", _fmt(state.feasibleRate, self.config.precision)),
            ("best value", _fmt(state.bestValue, self.config.precision)),
            ("bestX", _fmt_vector(state.bestX, self.config.precision, fullVectors=fullVectors)),
            ("meanX", _fmt_vector(state.meanX, self.config.precision, fullVectors=fullVectors)),
            ("stdX", _fmt_vector(state.stdX, self.config.precision, fullVectors=fullVectors)),
            ("elapsed", f"{state.elapsed:.1f}s"),
        ]
        return _renderBlock(f"[summary] {methodName}", pairs)

    def renderFinal(self, result, methodName):
        bestX, meanX, stdX = _result_vectors(result)
        pairs = [
            ("method", methodName),
            ("status", "finished"),
            ("iterations", result.iters),
            ("evaluations", result.FEs),
            ("chains", result.decs.shape[0] if result.decs.ndim == 3 else 0),
            ("draws", result.decs.shape[1] if result.decs.ndim == 3 else 0),
            ("mean logProb", _fmt(np.nanmean(result.logProb) if result.logProb.size else None, self.config.precision)),
            ("acceptance mean", _fmt(np.mean(result.acceptanceRate) if result.acceptanceRate.size else 0.0, self.config.precision)),
            ("acceptance min", _fmt(np.min(result.acceptanceRate) if result.acceptanceRate.size else 0.0, self.config.precision)),
            ("acceptance max", _fmt(np.max(result.acceptanceRate) if result.acceptanceRate.size else 0.0, self.config.precision)),
            ("feasible rate", _fmt(np.mean(result.feasibleMask) if result.feasibleMask.size else 0.0, self.config.precision)),
            ("best value", _fmt(_best_value(result.bestObjs), self.config.precision)),
            ("bestX", _fmt_vector(bestX, self.config.precision, fullVectors=False)),
            ("meanX", _fmt_vector(meanX, self.config.precision, fullVectors=False)),
            ("stdX", _fmt_vector(stdX, self.config.precision, fullVectors=False)),
            ("elapsed", f"{result.runtime:.1f}s"),
        ]
        return _renderBlock("Inference finished", pairs)

    def renderFinalFull(self, result, methodName):
        bestX, meanX, stdX = _result_vectors(result)
        pairs = [
            ("method", methodName),
            ("status", "finished"),
            ("iterations", result.iters),
            ("evaluations", result.FEs),
            ("chains", result.decs.shape[0] if result.decs.ndim == 3 else 0),
            ("draws", result.decs.shape[1] if result.decs.ndim == 3 else 0),
            ("mean logProb", _fmt(np.nanmean(result.logProb) if result.logProb.size else None, self.config.precision)),
            ("acceptance mean", _fmt(np.mean(result.acceptanceRate) if result.acceptanceRate.size else 0.0, self.config.precision)),
            ("acceptance min", _fmt(np.min(result.acceptanceRate) if result.acceptanceRate.size else 0.0, self.config.precision)),
            ("acceptance max", _fmt(np.max(result.acceptanceRate) if result.acceptanceRate.size else 0.0, self.config.precision)),
            ("feasible rate", _fmt(np.mean(result.feasibleMask) if result.feasibleMask.size else 0.0, self.config.precision)),
            ("best value", _fmt(_best_value(result.bestObjs), self.config.precision)),
            ("bestX", _fmt_vector(bestX, self.config.precision, fullVectors=True)),
            ("meanX", _fmt_vector(meanX, self.config.precision, fullVectors=True)),
            ("stdX", _fmt_vector(stdX, self.config.precision, fullVectors=True)),
            ("elapsed", f"{result.runtime:.1f}s"),
        ]
        return _renderBlock("Inference finished", pairs)


class VerboseReporter:
    def __init__(self, renderer, config, stream=None):
        self.renderer = renderer
        self.config = config
        self.stream = stream if stream is not None else sys.stdout
        self._lastProgressLength = 0
        self._useTTY = bool(config.useTTY and hasattr(self.stream, "isatty") and self.stream.isatty())

    def progress(self, state, methodName, problem):
        if self.config.level < 2 or state.iteration % self.config.progressEvery != 0:
            return
        text = self.renderer.renderProgress(state, methodName)
        if self._useTTY:
            padding = max(0, self._lastProgressLength - len(text))
            self._emit("\r" + text + (" " * padding), problem, end="")
            self._lastProgressLength = len(text)
        else:
            self._writeLine(text, problem)

    def final(self, result, methodName, problem):
        if self.config.level < 1:
            return
        self._newlineIfNeeded(problem)
        self._writeLine(self.renderer.renderFinal(result, methodName), problem)

    def summaryFull(self, state, methodName):
        return self.renderer.renderSummary(state, methodName, fullVectors=True)

    def finalFull(self, result, methodName):
        return self.renderer.renderFinalFull(result, methodName)

    def _writeLine(self, text, problem):
        self._emit(text, problem)
        self._lastProgressLength = 0

    def _newlineIfNeeded(self, problem):
        if self._useTTY and self._lastProgressLength > 0:
            self._emit("", problem)
            self._lastProgressLength = 0

    def _emit(self, text, problem, end="\n"):
        if self.stream is not None:
            self.stream.write(text + end)
            self.stream.flush()


class Verbose:
    workDir = os.getcwd()

    @staticmethod
    def setupContext(obj, problem):
        problem.verboseFlag = obj.verboseFlag
        problem.logLines = [] if obj.logFlag else None
        config = VerboseConfig(
            level=0 if not obj.verboseFlag else 2,
            progressEvery=max(1, obj.verboseFreq),
            summaryEvery=max(1, obj.verboseFreq),
            precision=4,
            useTTY=True,
        )
        stream = sys.stdout if obj.verboseFlag else None
        obj.reporter = VerboseReporter(InferenceRenderer(config), config, stream=stream)
        return obj.reporter

    @staticmethod
    def printSettings(obj):
        if not (obj.verboseFlag or getattr(obj.problem, "logLines", None) is not None):
            return
        lines = [
            f"Inference: {obj.name}",
            f"Problem: {obj.problem.name}",
            f"nInput: {obj.problem.nInput}",
            f"nOutput: {obj.problem.nOutput}",
            f"maxIters: {obj.maxIters}",
        ]
        for line in lines:
            obj.reporter._writeLine(line, obj.problem)

    @staticmethod
    def printIteration(obj):
        stateData = obj.state
        bestX, meanX, stdX = _state_vectors(stateData)
        state = ProgressState(
            iteration=obj.iters,
            nEval=obj.FEs,
            elapsed=stateData.runtime,
            meanLogProb=stateData.meanLogProb,
            currentLogProb=_current_log_prob(stateData.logProb),
            acceptanceRateMean=stateData.acceptanceRateMean,
            feasibleRate=stateData.feasibleRate,
            bestValue=stateData.bestObj,
            nChains=0 if stateData.decs is None else stateData.decs.shape[0],
            draws=0 if stateData.decs is None else stateData.decs.shape[1],
            acceptanceRate=stateData.acceptanceRate,
            bestX=bestX,
            meanX=meanX,
            stdX=stdX,
        )
        if obj.verboseFlag:
            obj.reporter.progress(state, obj.name, obj.problem)
        if obj.logFlag and obj.iters % obj.verboseFreq == 0:
            obj.problem.logLines.append(obj.reporter.summaryFull(state, obj.name) + "\n")

    @staticmethod
    def printConclusion(obj, result):
        if obj.verboseFlag:
            obj.reporter.final(result, obj.name, obj.problem)
        if obj.logFlag:
            obj.problem.logLines.append(obj.reporter.finalFull(result, obj.name) + "\n")

    @staticmethod
    def saveLog(obj):
        if not obj.logFlag:
            return
        problem = obj.problem
        workDir = problem.workDir if hasattr(problem, "workDir") else Verbose.workDir
        folder = os.path.join(workDir, "Result")
        os.makedirs(folder, exist_ok=True)
        runId = getattr(obj, "runId", None)
        if runId is None:
            timestamp = time.strftime("%Y%m%d_%H%M")
            runId = f"{obj.name.lower()}_{timestamp}_{os.getpid():x}"[-32:]
        filepath = os.path.join(folder, f"{runId}.log")
        with open(filepath, "w") as f:
            f.writelines(problem.logLines)


def _fmt(value, precision):
    if value is None:
        return "-"
    if abs(float(value)) == 0:
        return "0"
    return f"{float(value):.{precision}e}"


def _best_value(bestObjs):
    if bestObjs is None:
        return None
    return float(np.asarray(bestObjs).reshape(-1)[0])


def _current_log_prob(logProb):
    if logProb is None or logProb.size == 0:
        return None
    return float(np.nanmean(logProb[:, -1]))


def _state_vectors(state):
    if state.decs is None or state.decs.size == 0:
        return None, None, None
    flatDecs = state.decs.reshape(-1, state.decs.shape[-1])
    flatFeasible = state.feasibleMask.reshape(-1) if state.feasibleMask is not None else np.ones(flatDecs.shape[0], dtype=bool)
    sample = flatDecs[flatFeasible] if np.any(flatFeasible) else flatDecs
    bestX = None if state.bestDecs is None else np.asarray(state.bestDecs).reshape(-1)
    return bestX, np.mean(sample, axis=0), np.std(sample, axis=0)


def _result_vectors(result):
    if result.decs.size == 0:
        return None, None, None
    flatDecs = result.decs.reshape(-1, result.decs.shape[-1])
    flatFeasible = result.feasibleMask.reshape(-1) if result.feasibleMask is not None else np.ones(flatDecs.shape[0], dtype=bool)
    sample = flatDecs[flatFeasible] if np.any(flatFeasible) else flatDecs
    bestX = None if result.bestDecs is None else np.asarray(result.bestDecs).reshape(-1)
    return bestX, np.mean(sample, axis=0), np.std(sample, axis=0)


def _fmt_vector(array, precision, previewSize=8, fullVectors=False):
    if array is None:
        return "-"
    vec = np.asarray(array).reshape(-1)
    shown = vec if fullVectors else vec[:previewSize]
    parts = [_fmt(item, precision) for item in shown]
    if not fullVectors and vec.size > previewSize:
        parts.append("...")
    return "[" + ", ".join(parts) + "]"


def _nan_min(array):
    if array is None or np.asarray(array).size == 0:
        return None
    return float(np.nanmin(array))


def _nan_max(array):
    if array is None or np.asarray(array).size == 0:
        return None
    return float(np.nanmax(array))


def _renderBlock(title, pairs):
    width = max(len(key) for key, _ in pairs)
    lines = [title]
    lines.extend(f"  {key.ljust(width)} : {value}" for key, value in pairs)
    return "\n".join(lines)
