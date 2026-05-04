from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class VerboseConfig:
    precision: int = 4
    showParams: bool = True
    maxMetricTerms: int = 5


class VerboseReporter:
    def __init__(self, config: VerboseConfig, stream=None):
        self.config = config
        self.stream = stream if stream is not None else sys.stdout

    def writeLine(self, text: str):
        if self.stream is None:
            return
        self.stream.write(text + "\n")
        self.stream.flush()


class Verbose:
    workDir = os.getcwd()

    @staticmethod
    def setupContext(obj, problem):
        problem.verboseFlag = obj.verboseFlag
        problem.logLines = [] if obj.logFlag else None
        stream = sys.stdout if obj.verboseFlag else None
        obj.reporter = VerboseReporter(VerboseConfig(), stream=stream)
        return obj.reporter

    @staticmethod
    def printSettings(obj):
        if not (obj.verboseFlag or obj.problem.logLines is not None):
            return

        lines = [
            f"Analysis: {obj.name}",
            f"Problem: {obj.problem.name}",
            f"nInput: {obj.problem.nInput}",
            f"nOutput: {obj.problem.nOutput}",
        ]
        runId = getattr(obj, "runId", None)
        if runId is not None:
            lines.append(f"runId: {runId}")
        if obj.reporter.config.showParams and obj.setting.asDict():
            lines.append(f"params: {obj.setting.asDict()}")

        for line in lines:
            Verbose._emit(obj, line)

    @staticmethod
    def printConclusion(obj, result):
        if not (obj.verboseFlag or obj.problem.logLines is not None):
            return

        Verbose._emitConsole(obj, "Analysis finished")
        Verbose._emitConsole(obj, f"runtime: {result.runtime:.3f}s")
        Verbose._emitLog(obj, "Analysis finished")
        Verbose._emitLog(obj, f"runtime: {result.runtime:.3f}s")
        Verbose._emitLog(obj, f"target: {result.target}")
        if result.meta:
            Verbose._emitLog(obj, f"meta: {result.meta}")
        for metric in result.metrics:
            Verbose._emitConsole(obj, f"[{metric.name}]")
            for rowLabel, rowValues in zip(metric.rowLabels, metric.values):
                Verbose._emitConsole(
                    obj,
                    _format_top_terms(
                        rowLabel,
                        metric.colLabels,
                        rowValues,
                        obj.reporter.config.precision,
                        obj.reporter.config.maxMetricTerms,
                    ),
                )

            Verbose._emitLog(obj, f"[{metric.name}]")
            for line in _format_full_metric(metric, obj.reporter.config.precision):
                Verbose._emitLog(obj, line)

    @staticmethod
    def saveLog(obj):
        if not obj.logFlag:
            return
        problem = obj.problem
        workDir = problem.workDir if hasattr(problem, "workDir") else Verbose.workDir
        folder = Verbose.checkDir(workDir)
        runId = getattr(obj, "runId", None)
        if runId is None:
            timestamp = time.strftime("%Y%m%d_%H%M")
            runId = f"{obj.name.lower()}_{timestamp}_{os.getpid():x}"[-32:]
        filepath = os.path.join(folder, f"{runId}.log")
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(problem.logLines)

    @staticmethod
    def checkDir(workDir):
        folder = os.path.join(workDir, "Result")
        os.makedirs(folder, exist_ok=True)
        return folder

    @staticmethod
    def _emit(obj, text):
        if obj.verboseFlag:
            obj.reporter.writeLine(text)
        if obj.problem.logLines is not None:
            obj.problem.logLines.append(text + "\n")

    @staticmethod
    def _emitConsole(obj, text):
        if obj.verboseFlag:
            obj.reporter.writeLine(text)

    @staticmethod
    def _emitLog(obj, text):
        if obj.problem.logLines is not None:
            obj.problem.logLines.append(text + "\n")

def _format_value(value, precision: int):
    value = float(value)
    if value == 0.0:
        return "0"
    return f"{value:.{precision}e}"


def _format_top_terms(rowLabel, colLabels, rowValues, precision: int, maxTerms: int):
    indexed = list(zip(colLabels, np.asarray(rowValues, dtype=float)))
    indexed.sort(key=lambda item: abs(item[1]), reverse=True)
    top = indexed[:maxTerms]
    pairs = [f"{label}={_format_value(value, precision)}" for label, value in top]
    suffix = ""
    if len(indexed) > maxTerms:
        suffix = f"  ... ({len(indexed) - maxTerms} more)"
    return f"{rowLabel}: " + "  ".join(pairs) + suffix


def _format_full_metric(metric, precision: int):
    lines = ["columns: " + "  ".join(str(label) for label in metric.colLabels)]
    for rowLabel, rowValues in zip(metric.rowLabels, metric.values):
        values = "  ".join(_format_value(value, precision) for value in np.asarray(rowValues, dtype=float))
        lines.append(f"{rowLabel}: {values}")
    return lines
