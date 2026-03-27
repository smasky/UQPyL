import os
import shlex
import shutil
import subprocess
import queue
import numpy as np
import threading
import atexit
import signal

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from UQPyL.problem import Problem

from param_manager import ParamManager
from series_extractor import SeriesExtractor
from evaluator import Evaluator
from func_manager import FunctionManager
from load_cfg_general import load_config
from run_reporter import RunReporter
from run_error import RunError

class SimModel(Problem):
    def __init__(self, cfgPath: str):
        self.cfgPath = cfgPath
        self.cfg = load_config(cfgPath)

        # Create run queue / backup path
        self.create_run_queue()

        # Managers
        self.functionManager = FunctionManager(self.cfg)
        self.paramManager = ParamManager(self.cfg, self.functionManager, self.backupPath)
        self.seriesExtractor = SeriesExtractor(self.cfg, self.functionManager)
        self.evaluator = Evaluator(self.cfg, self.functionManager)

        # For UQPyL
        nInput, xLabels, varType, varSet, ub, lb = self.paramManager.get_param_info()
        nOutput, optType, nConstraints = self.evaluator.get_evaluation_info()

        # For Debug
        self.reporter = RunReporter(self.backupPath, xLabels, self.cfg)
        self.reporter.start()

        self._closed = False
        self._error_log_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._cleanup_done = False

        # register exit hooks
        atexit.register(self._cleanup_on_exit)

        try:
            signal.signal(signal.SIGINT, self._handle_termination)
            signal.signal(signal.SIGTERM, self._handle_termination)
        except Exception:
            # signal registration may fail in non-main thread / some environments
            pass
        
        super().__init__(
            nInput=nInput,
            nOutput=nOutput,
            varType=varType,
            varSet=varSet,
            ub=ub,
            lb=lb,
            xLabels=xLabels,
            optType=optType,
        )

    def create_run_queue(self):
        
        now = datetime.now()
        base_time_str = now.strftime("%m%d_%H%M%S")

        self.runPath = os.path.join(self.cfg.basic.workPath, "tempRun", base_time_str)

     
        counter = 0
        while os.path.exists(self.runPath):
         
            micro_str = f"{now.microsecond + counter:06d}"
            self.runPath = os.path.join(self.cfg.basic.workPath, "tempRun", base_time_str + "_" + micro_str)
            counter += 1
 
        os.makedirs(self.runPath, exist_ok=True)

        self.runQueue = queue.Queue()

        for i in range(self.cfg.basic.parallel):
            path = os.path.join(self.runPath, f"instance_{i}")
            shutil.copytree(self.cfg.basic.projectPath, path)
            self.runQueue.put(path)

        self.backupPath = os.path.join(self.runPath, "backup")
        os.makedirs(self.backupPath, exist_ok=True)

        shutil.copy(self.cfg.parameters.library, self.backupPath)
        shutil.copy(self.cfg.parameters.params, self.backupPath)
        shutil.copy(self.cfgPath, self.backupPath)

        for series in self.cfg.series:
            if series.obs:
                obs_src = os.path.join(self.cfg.basic.workPath, series.obs.file)
                shutil.copy(obs_src, self.backupPath)

    def _build_command(self, workPath):
        raw_cmd = self.cfg.basic.command

        if isinstance(raw_cmd, (list, tuple)):
            cmd = list(raw_cmd)
        else:
            cmd = shlex.split(str(raw_cmd))

        if not cmd:
            raise ValueError("cfg.basic.command is empty")

        # First token as executable/script path
        if not os.path.isabs(cmd[0]):
            cmd[0] = os.path.join(workPath, cmd[0])

        return cmd

    def _to_float_or_nan(self, value):
        if value is None:
            return np.nan

        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:
                pass

        arr = np.asarray(value)

        if arr.size == 0:
            return np.nan

        if arr.size == 1:
            return float(arr.reshape(-1)[0])

        return np.nan

    def _objective_penalty(self, j):
        # keep consistent with your current NaN handling
        return np.inf * self.opt[j]

    def _constraint_penalty(self):
        # assume g(x) <= 0 ; infeasible / failed => +inf
        return np.inf

    def evaluate(self, X):
        n = X.shape[0]
        n_obj = self.evaluator.nOutput
        n_con = self.evaluator.nConstraints

        objs = np.zeros((n, n_obj))
        cons = np.full((n, n_con), self._constraint_penalty()) if n_con > 0 else None

        batch_id = self.reporter.new_batch_id()
        records = []

        if self.cfg.basic.parallel > 1:
            with ThreadPoolExecutor(max_workers=self.cfg.basic.parallel) as executor:
                futures = [
                    executor.submit(self._subprocess, X[i, :], i, batch_id)
                    for i in range(n)
                ]
                records = [future.result() for future in futures]
        else:
            for i in range(n):
                records.append(self._subprocess(X[i, :], i, batch_id))

        for rec in records:
            i = int(rec["i"])
            
            for j, obj_id in enumerate(self.cfg.objectives.use):
                val = self._to_float_or_nan(rec.get(obj_id, np.nan))
                
                # TODO
                if np.isnan(val):
                    val = self._objective_penalty(j)
                    
                objs[i, j] = val

            if cons is not None:
                for j, con_id in enumerate(self.cfg.constraints.use):
                    val = self._to_float_or_nan(rec.get(con_id, np.nan))
                    
                    # TODO
                    if np.isnan(val):
                        val = self._constraint_penalty()

                    cons[i, j] = val

        return {"objs": objs, "cons": cons}

    def objFunc(self, X):
        res = self.evaluate(X)
        return res["objs"]

    def conFunc(self, X):
        res = self.evaluate(X)
        return res["cons"]

    def _subprocess(self, X, i, batch_id):
        workPath = self.runQueue.get()

        context = {
            "X": np.asarray(X).ravel(),
            "i": int(i),
            "batch_id": int(batch_id),
        }

        stdout = ""
        stderr = ""

        try:
            
            # Write parameter values to runPath
            self.paramManager.set_values(workPath, X, context)
            context.update(self.paramManager.get_cached_env(X))

            # Run
            cmd = self._build_command(workPath)
            process = subprocess.Popen(
                cmd,
                cwd=workPath,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            timeout = self.cfg.basic.timeout if self.cfg.basic.timeout > 0 else None

            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise RunError(
                stage="subprocess",
                code="TIMEOUT",
                target="simulation",
                message=f"Simulation timed out after {timeout}s"
                )

            if process.returncode != 0:
                stderr_tail = stderr[-500:] if stderr else ""
                raise RunError(
                    stage="subprocess",
                    code="NONZERO_EXIT",
                    target="simulation",
                    message=f"Return code {process.returncode}; stderr={stderr_tail}"
                )
        
            # extract series
            context = self.seriesExtractor.extract_all(workPath, context)
            
            # evaluate
            scalars = self.evaluator.evaluate_all(context)
            
            context.update(scalars)

        except RunError as e:
            context["error"] = {
                "stage": e.stage,
                "code": e.code,
                "target": e.target,
                "message": e.message,
            }

            self._write_error_log(batch_id, i, e)
            
        except Exception as e:
            unknown_err = RunError(
                stage="unknown",
                code="UNEXPECTED_EXCEPTION",
                target="simulation",
                message=str(e),
            )

            context["error"] = {
                "stage": unknown_err.stage,
                "code": unknown_err.code,
                "target": unknown_err.target,
                "message": unknown_err.message,
            }
            
            self._write_error_log(batch_id, i, unknown_err)

        finally:
            self.runQueue.put(workPath)

            if "error" in context:
                self._error_handler(context, self.cfg)
            
            if hasattr(self, "reporter") and self.reporter is not None:
                try:
                    self.reporter.submit(context)
                except RuntimeError:
                    pass

        return context

    # error
    def _error_handler(self, context, cfg):
        
        for obj_id in cfg.objectives.use:
            context[obj_id] = cfg.objectives.items[obj_id].on_error
        for con_id in cfg.constraints.use:
            context[con_id] = cfg.constraints.items[con_id].on_error
        for diag_id in cfg.diagnostics.use:
            context[diag_id] = cfg.diagnostics.items[diag_id].on_error
        
    def _format_error_line(self, batch_id, run_id, err: RunError):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[{timestamp}] "
            f"batch={batch_id} "
            f"run={run_id} "
            f"level=ERROR "
            f"stage={err.stage} "
            f"code={err.code} "
            f"target={err.target} "
            f"message={err.message}"
        )

    def _write_error_log(self, batch_id, run_id, err: RunError):
        
        error_file = os.path.join(self.backupPath, "error.txt")
        line = self._format_error_line(batch_id, run_id, err)

        with self._error_log_lock:
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    
    # close and exit
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if getattr(self, "_closed", False):
            return

        self._closed = True

        if hasattr(self, "reporter") and self.reporter is not None:
            try:
                self.reporter.close()
            except Exception as e:
                print(f"[SimModel.close] reporter.close() failed: {e}")

        try:
            self._cleanup_instances()
        except Exception as e:
            print(f"[SimModel.close] instance cleanup failed: {e}")
            print("Please clean instance folders manually if needed.")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    def _cleanup_instances(self):
        """
        Remove instance_* folders under runPath, keep backup folder.
        Best-effort cleanup only.
        """
        if not hasattr(self, "runPath") or not self.runPath:
            return

        with self._cleanup_lock:
            if self._cleanup_done:
                return

            failed = []

            try:
                for name in os.listdir(self.runPath):
                    full_path = os.path.join(self.runPath, name)

                    if not os.path.isdir(full_path):
                        continue

                    if name == "backup":
                        continue

                    if name.startswith("instance_"):
                        try:
                            shutil.rmtree(full_path)
                        except Exception as e:
                            failed.append((full_path, str(e)))

            finally:
                self._cleanup_done = True

            if failed:
                print("[SimModel.cleanup] Failed to remove some instance folders:")
                for path, err in failed:
                    print(f"  - {path}: {err}")
                print("Please remove them manually if needed.")
    
    def _cleanup_on_exit(self):
        try:
            self.close()
        except Exception as e:
            print(f"[SimModel._cleanup_on_exit] cleanup failed: {e}")


    def _handle_termination(self, signum, frame):
        """
        Best-effort cleanup for SIGINT / SIGTERM.
        Cannot handle SIGKILL.
        """
        try:
            print(f"[SimModel] Received signal {signum}, cleaning instance folders...")
            self.close()
        except Exception as e:
            print(f"[SimModel._handle_termination] cleanup failed: {e}")

        raise SystemExit(128 + signum)