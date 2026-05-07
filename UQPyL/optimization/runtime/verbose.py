import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.runtime import ensure_result_dir

@dataclass
class VerboseConfig:
    level: int = 2
    progressEvery: int = 1
    summaryEvery: int = 50
    precision: int = 4
    maxParetoPreview: int = 5
    showParams: bool = False
    useTTY: bool = True


@dataclass
class ProgressState:
    iteration: int
    nEval: int
    elapsed: float
    status: str = "running"
    bestValue: float | None = None
    bestX: np.ndarray | None = None
    paretoSize: int | None = None
    paretoPreview: np.ndarray | None = None
    hypervolume: float | None = None
    constraintViolation: float | None = None
    params: dict[str, Any] | None = None


class SingleObjectiveRenderer:
    def __init__(self, config: VerboseConfig):
        self.config = config

    def renderProgress(self, state: ProgressState, algorithmName: str) -> str:
        return (
            f"{algorithmName} | iter={state.iteration} "
            f"eval={state.nEval} best={_fmt(state.bestValue, self.config.precision)} "
            f"cv={_fmt(state.constraintViolation, self.config.precision)} "
            f"time={state.elapsed:.1f}s"
        )

    def renderSummary(self, state: ProgressState, algorithmName: str) -> str:
        pairs = [
            ("iter", state.iteration),
            ("evaluations", state.nEval),
            ("best value", _fmt(state.bestValue, self.config.precision)),
            ("best X", _fmt_vector(state.bestX, self.config.precision, 8)),
            ("constraint viol.", _fmt(state.constraintViolation, self.config.precision)),
            ("elapsed", f"{state.elapsed:.1f}s"),
        ]
        return _renderBlock(f"[summary] {algorithmName}", pairs)

    def renderSummaryFull(self, state: ProgressState, algorithmName: str) -> str:
        return self.renderSummary(state, algorithmName)

    def renderFinal(self, result: Any, algorithmName: str) -> str:
        bestValue = None if result.bestObjs is None else float(result.bestObjs[0, 0])
        pairs = [
            ("algorithm", algorithmName),
            ("status", "finished"),
            ("iterations", result.iters),
            ("evaluations", result.FEs),
            ("best value", _fmt(bestValue, self.config.precision)),
            ("best X", _fmt_vector(result.bestDecs, self.config.precision, 8)),
            ("constraint viol.", _fmt(_constraint_violation(result.bestCons), self.config.precision)),
            ("elapsed", f"{result.runtime:.1f}s"),
        ]
        return _renderBlock("Optimization finished", pairs)

    def renderFinalFull(self, result: Any, algorithmName: str) -> str:
        return self.renderFinal(result, algorithmName)


class MultiObjectiveRenderer:
    def __init__(self, config: VerboseConfig):
        self.config = config

    def renderProgress(self, state: ProgressState, algorithmName: str) -> str:
        return (
            f"{algorithmName} | iter={state.iteration} "
            f"eval={state.nEval} nd={state.paretoSize if state.paretoSize is not None else '-'} "
            f"hv={_fmt(state.hypervolume, self.config.precision)} "
            f"cv={_fmt(state.constraintViolation, self.config.precision)} "
            f"time={state.elapsed:.1f}s"
        )

    def renderSummary(self, state: ProgressState, algorithmName: str) -> str:
        pairs = [
            ("iter", state.iteration),
            ("evaluations", state.nEval),
            ("pareto size", state.paretoSize if state.paretoSize is not None else "-"),
            ("hypervolume", _fmt(state.hypervolume, self.config.precision)),
        ]
        lines = [_renderBlock(f"[summary] {algorithmName}", pairs), "  pareto preview  :"]
        lines.extend(_fmt_pareto_preview(state.paretoPreview, self.config.precision, self.config.maxParetoPreview))
        trailing = [
            ("constraint viol.", _fmt(state.constraintViolation, self.config.precision)),
            ("elapsed", f"{state.elapsed:.1f}s"),
        ]
        lines.extend(_renderPairs(trailing))
        return "\n".join(lines)

    def renderSummaryFull(self, state: ProgressState, algorithmName: str) -> str:
        pairs = [
            ("iter", state.iteration),
            ("evaluations", state.nEval),
            ("pareto size", state.paretoSize if state.paretoSize is not None else "-"),
            ("hypervolume", _fmt(state.hypervolume, self.config.precision)),
            ("constraint viol.", _fmt(state.constraintViolation, self.config.precision)),
            ("elapsed", f"{state.elapsed:.1f}s"),
        ]
        lines = [_renderBlock(f"[summary] {algorithmName}", pairs), "  pareto:"]
        lines.extend(_fmt_pareto_preview(state.paretoPreview, self.config.precision, 10))
        return "\n".join(lines)

    def renderFinal(self, result: Any, algorithmName: str) -> str:
        pairs = [
            ("algorithm", algorithmName),
            ("status", "finished"),
            ("iterations", result.iters),
            ("evaluations", result.FEs),
            ("pareto size", result.bestObjs.shape[0] if result.bestObjs is not None else 0),
            ("hypervolume", _fmt(result.bestMetric, self.config.precision)),
            ("constraint viol.", _fmt(_constraint_violation(result.bestCons), self.config.precision)),
            ("elapsed", f"{result.runtime:.1f}s"),
        ]
        lines = [_renderBlock("Multi-objective optimization finished", pairs), "", "  Pareto preview:"]
        lines.extend(_fmt_pareto_preview(result.bestObjs, self.config.precision, self.config.maxParetoPreview, indent="    "))
        return "\n".join(lines)

    def renderFinalFull(self, result: Any, algorithmName: str) -> str:
        pairs = [
            ("algorithm", algorithmName),
            ("status", "finished"),
            ("iterations", result.iters),
            ("evaluations", result.FEs),
            ("pareto size", result.bestObjs.shape[0] if result.bestObjs is not None else 0),
            ("hypervolume", _fmt(result.bestMetric, self.config.precision)),
            ("constraint viol.", _fmt(_constraint_violation(result.bestCons), self.config.precision)),
            ("elapsed", f"{result.runtime:.1f}s"),
        ]
        lines = [_renderBlock("Multi-objective optimization finished", pairs), "", "  pareto:"]
        lines.extend(_fmt_pareto_preview(result.bestObjs, self.config.precision, 1000000, indent="    "))
        return "\n".join(lines)


class VerboseReporter:
    def __init__(self, renderer, config: VerboseConfig, stream=None):
        self.renderer = renderer
        self.config = config
        self.stream = stream if stream is not None else sys.stdout
        self._lastProgressLength = 0
        self._useTTY = bool(config.useTTY and hasattr(self.stream, "isatty") and self.stream.isatty())

    def shouldProgress(self, iteration: int) -> bool:
        if self.config.level < 2:
            return False
        interval = self.config.progressEvery if self._useTTY else max(self.config.progressEvery, 10)
        return iteration % interval == 0

    def shouldSummary(self, iteration: int) -> bool:
        return self.config.level >= 3 and iteration % self.config.summaryEvery == 0

    def progress(self, state: ProgressState, algorithmName: str, problem):
        if not self.shouldProgress(state.iteration):
            return
        text = self.renderer.renderProgress(state, algorithmName)
        self._writeProgress(text, problem)

    def summary(self, state: ProgressState, algorithmName: str, problem):
        if not self.shouldSummary(state.iteration):
            return
        self._newlineIfNeeded(problem)
        text = self.renderer.renderSummary(state, algorithmName)
        self._writeLine(text, problem)

    def final(self, result: Any, algorithmName: str, problem):
        if self.config.level < 1:
            return
        self._newlineIfNeeded(problem)
        text = self.renderer.renderFinal(result, algorithmName)
        self._writeLine(text, problem)

    def _writeProgress(self, text: str, problem):
        if self._useTTY:
            padding = max(0, self._lastProgressLength - len(text))
            line = "\r" + text + (" " * padding)
            self._emit(line, problem, end="")
            self._lastProgressLength = len(text)
        else:
            self._writeLine(text, problem)

    def _writeLine(self, text: str, problem):
        self._emit(text, problem)
        self._lastProgressLength = 0

    def _newlineIfNeeded(self, problem):
        if self._useTTY and self._lastProgressLength > 0:
            self._emit("", problem)
            self._lastProgressLength = 0

    def _emit(self, text: str, problem, end="\n"):
        if self.stream is not None:
            self.stream.write(text + end)
            self.stream.flush()


class Verbose:
    workDir = os.getcwd()

    @staticmethod
    def _resolveRunId(obj):
        session = getattr(obj, "session", None)
        if session is not None:
            run_id = getattr(session, "run_id", None)
            if run_id is not None:
                return run_id
        runId = getattr(obj, "runId", None)
        if runId is not None:
            return runId
        timestamp = time.strftime("%Y%m%d_%H%M")
        return f"{obj.name.lower()}_{timestamp}_{os.getpid():x}"[-32:]

    @staticmethod
    def makeReporter(nObj: int, config: VerboseConfig | None = None, stream=None):
        cfg = config or VerboseConfig()
        renderer = SingleObjectiveRenderer(cfg) if nObj == 1 else MultiObjectiveRenderer(cfg)
        return VerboseReporter(renderer, cfg, stream=stream)

    @staticmethod
    def setupContext(obj, problem):
        problem.verboseFlag = obj.verboseFlag
        problem.logLines = [] if obj.logFlag else None

        config = VerboseConfig(
            level=0 if not obj.verboseFlag else 2,
            progressEvery=1,
            summaryEvery=max(1, obj.verboseFreq),
            precision=4,
            maxParetoPreview=5,
            showParams=False,
            useTTY=True,
        )
        stream = sys.stdout if obj.verboseFlag else None
        obj.reporter = Verbose.makeReporter(problem.nObj, config=config, stream=stream)
        return obj.reporter

    @staticmethod
    def printSettings(obj):
        if not (obj.verboseFlag or obj.problem.logLines):
            return
        lines = [
            f"Algorithm: {obj.name}",
            f"Problem: {obj.problem.name}",
            f"nInput: {obj.problem.nInput}",
            f"nObj: {obj.problem.nObj}",
            f"maxFEs: {obj.maxFEs}",
            f"maxIters: {obj.maxIter}",
        ]
        if obj.reporter.config.showParams:
            lines.append(f"params: {obj.params.asDict()}")
        for line in lines:
            obj.reporter._writeLine(line, obj.problem)

    @staticmethod
    def printIteration(obj):
        state = ProgressState(
            iteration=obj.iters,
            nEval=obj.FEs,
            elapsed=obj.state.runtime,
            bestValue=_single_best_value(obj.state.bestObjs),
            bestX=obj.state.bestDecs,
            paretoSize=None if obj.problem.nObj == 1 or obj.state.bestObjs is None else obj.state.bestObjs.shape[0],
            paretoPreview=None if obj.problem.nObj == 1 else obj.state.bestObjs,
            hypervolume=obj.state.bestMetric,
            constraintViolation=_constraint_violation(obj.state.bestCons),
        )
        if obj.verboseFlag:
            obj.reporter.progress(state, obj.name, obj.problem)
            obj.reporter.summary(state, obj.name, obj.problem)
        if obj.logFlag and obj.iters % obj.verboseFreq == 0:
            fullText = obj.reporter.renderer.renderSummaryFull(state, obj.name)
            obj.problem.logLines.append(fullText + "\n")

    @staticmethod
    def printConclusion(obj, result):
        if obj.verboseFlag:
            obj.reporter.final(result, obj.name, obj.problem)
        if obj.logFlag:
            fullText = obj.reporter.renderer.renderFinalFull(result, obj.name)
            obj.problem.logLines.append(fullText + "\n")

    @staticmethod
    def saveToNPZ(filepath, res):
        np.savez_compressed(filepath, **res)

    @staticmethod
    def checkDir(workDir):
        return ensure_result_dir(workDir)

    @staticmethod
    def saveData(obj, resultData):
        problem = obj.problem
        workDir = problem.workDir if hasattr(problem, "GUI") else Verbose.workDir
        folder = Verbose.checkDir(workDir)
        runId = Verbose._resolveRunId(obj)
        filepath = os.path.join(folder, f"{runId}.npz")
        Verbose.saveToNPZ(filepath, resultData)

    @staticmethod
    def saveLog(obj):
        if not obj.logFlag:
            return
        problem = obj.problem
        workDir = problem.workDir if hasattr(problem, "GUI") else Verbose.workDir
        folder = Verbose.checkDir(workDir)
        runId = Verbose._resolveRunId(obj)
        filepath = os.path.join(folder, f"{runId}.log")
        with open(filepath, "w") as f:
            f.writelines(problem.logLines)


def _fmt(value, precision: int):
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if abs(float(value)) == 0:
        return "0"
    return f"{float(value):.{precision}e}"


def _fmt_vector(array, precision: int, previewSize: int):
    if array is None:
        return "-"
    vec = np.asarray(array).reshape(-1)
    parts = [f"{float(item):.{precision}e}" for item in vec[:previewSize]]
    if vec.size > previewSize:
        parts.append("...")
    return "[" + ", ".join(parts) + "]"


def _fmt_pareto_preview(objs, precision: int, previewCount: int, indent="    "):
    if objs is None:
        return [indent + "-"]
    arr = np.asarray(objs)
    lines = []
    for idx, row in enumerate(arr[:previewCount]):
        lines.append(f"{indent}{idx}: {_fmt_vector(row, precision, 8)}")
    return lines


def _renderBlock(title: str, pairs: list[tuple[str, Any]]):
    lines = [title]
    lines.extend(_renderPairs(pairs))
    return "\n".join(lines)


def _renderPairs(pairs: list[tuple[str, Any]]):
    if not pairs:
        return []
    width = max(len(key) for key, _ in pairs)
    return [f"  {key.ljust(width)} : {value}" for key, value in pairs]


def _constraint_violation(cons):
    if cons is None:
        return 0.0
    cons = np.asarray(cons)
    return float(np.sum(np.maximum(0.0, cons)))


def _single_best_value(bestObjs):
    if bestObjs is None:
        return None
    return float(np.asarray(bestObjs).reshape(-1)[0])
