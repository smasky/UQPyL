"""Resource ownership for public run/analyze calls, including subclass entries."""

import functools
import time
import warnings


def recordCleanupError(error, stage, cleanupError):
    note = f"Run cleanup failed during {stage}: {type(cleanupError).__name__}: {cleanupError}"
    if hasattr(error, "add_note"):
        error.add_note(note)
    else:  # Python 3.10 does not display exception notes.
        error.__notes__ = [*getattr(error, "__notes__", []), note]
        try:
            warnings.warn(note, RuntimeWarning, stacklevel=2)
        except BaseException:
            pass  # Warning filters must not replace the original exception.


class RunLifecycle:
    _runEntryPoint = "run"

    def _startRun(self):
        from .runtime import make_run_id
        self.runId = make_run_id(getattr(self, 'name', type(self).__name__), self.problem.name)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        method = cls.__dict__.get(cls._runEntryPoint)
        if method is None or getattr(method, "__isabstractmethod__", False):
            return

        @functools.wraps(method)
        def managedRun(self, *args, **kwargs):
            # A subclass delegating to super shares the outer call's lifetime.
            if getattr(self, "_runActive", False):
                return method(self, *args, **kwargs)
            if self.session is not None:
                raise RuntimeError("Close the existing run session before starting another run.")
            self._runActive = True
            self.runId = None
            start = time.perf_counter()
            try:
                result = method(self, *args, **kwargs)
                self._closeRunSession()
                return result
            except BaseException as error:
                session = self.session
                if session is not None:
                    self.storage.abort_run(
                        session, error, runtime=time.perf_counter() - start,
                        final_fes=getattr(self, "FEs", None),
                        final_iters=getattr(self, "iters", None),
                    )
                raise
            finally:
                if self.session is not None and self.session.conn is None:
                    self.session = None
                self._runActive = False

        setattr(cls, cls._runEntryPoint, managedRun)

    def _closeRunSession(self):
        if self.session is not None:
            self.storage.close(self.session)
            self.session = None

    def _closeStandaloneSession(self):
        if not getattr(self, "_runActive", False):
            self._closeRunSession()
