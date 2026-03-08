from extract_handler import read_extract
from load_cfg_general import ExprSpec, CallSpec

from run_error import RunError

class SeriesExtractor:
    def __init__(self, cfg, funcManager):
        self.cfg = cfg
        self.funcManager = funcManager

        required_sids = self._parse_required_sids()
        self.seriesDict = self._init_series(required_sids)

    def _parse_required_sids(self) -> set:
        required_sids = set()

        for d in self.cfg.derived:
            if hasattr(d, "call") and d.call is not None:
                for _, v in d.call.args.items():
                    if isinstance(v, str) and (v.endswith("_sim") or v.endswith("_obs")):
                        required_sids.add(v.replace("_sim", "").replace("_obs", ""))

            if hasattr(d, "expr") and d.expr is not None:
                for dep in d.expr.deps:
                    if isinstance(dep, str) and (dep.endswith("_sim") or dep.endswith("_obs")):
                        required_sids.add(dep.replace("_sim", "").replace("_obs", ""))

        for s in self.cfg.series:
            if getattr(s, "cache", False):
                required_sids.add(s.id)

        return required_sids

    def _init_series(self, cache_sids: set) -> dict:
        seriesDict = {}

        for s in self.cfg.series:
            if s.id not in cache_sids:
                continue

            series_id = s.id
            obsItem = s.obs
            simItem = s.sim

            obsData = None
            if obsItem:
                obsData = read_extract(None, obsItem)

            seriesDict[series_id] = {
                "obs": obsData,
                "simItem": simItem,
                "obsItem": obsItem,
            }

        return seriesDict

    def _resolve_series_ref(self, ref: str, workPath: str, env: dict):
        if ref in env:
            return env[ref]

        is_obs = ref.endswith("_obs")
        base_id = ref.replace("_sim", "").replace("_obs", "")

        if base_id not in self.cfg.series_index:
            raise RunError(
                stage="series",
                code="DEPENDENCY_MISSING",
                target=ref,
                message=f"Unknown series dependency '{ref}'"
            )

        if is_obs:
            if base_id not in self.seriesDict:
                raise RunError(
                    stage="series",
                    code="OBS_NOT_INITIALIZED",
                    target=ref,
                    message=f"Obs dependency '{ref}' is not initialized in seriesDict"
                )

            obs_data = self.seriesDict[base_id]["obs"]
            if obs_data is None:
                raise RunError(
                    stage="series",
                    code="MISSING_OBS_FILE",
                    target=ref,
                    message=f"Dependency '{ref}' requested, but '{base_id}' has no obs file"
                )

            env[ref] = obs_data
        else:
            sim_item = self.cfg.series_index[base_id].sim
            try:
                env[ref] = read_extract(workPath, sim_item)
            except Exception as e:
                raise RunError(
                    stage="series",
                    code="FILE_READ_ERROR",
                    target=ref,
                    message=f"Error reading simulation file for '{base_id}': {e}"
                )

        return env[ref]

    def extract_all(self, workPath: str, context: dict) -> dict:
        env = context.copy()

        for sid, item in self.seriesDict.items():
            simItem = item["simItem"]

            if isinstance(simItem, ExprSpec):
                for dep in simItem.deps:
                    if dep not in env:
                        self._resolve_series_ref(dep, workPath, env)

                try:
                    env[f"{sid}_sim"] = eval(simItem.expr, {"__builtins__": {}}, env)

                    if item["obs"] is not None:
                        env[f"{sid}_obs"] = item["obs"]

                except NameError as e:
                    raise RunError(
                        stage="series",
                        code="EXPR_EVAL_FAILED",
                        target=sid,
                        message=f"Expr evaluation failed for series '{sid}'. Missing dependency: {e}"
                    ) from e
                except Exception as e:
                    raise RunError(
                        stage="series",
                        code="EXPR_EVAL_FAILED",
                        target=sid,
                        message=f"Expr evaluation failed for series '{sid}': {e}"
                    ) from e

            elif isinstance(simItem, CallSpec):
                func_name = simItem.func
                raw_args = simItem.args

                func_args = {}
                for arg_k, arg_v in raw_args.items():
                    if arg_v not in env:
                        self._resolve_series_ref(arg_v, workPath, env)
                    func_args[arg_k] = env[arg_v]

                try:
                    func_result = self.funcManager.call(func_name, **func_args)
                    env[f"{sid}_sim"] = func_result
                except Exception as e:
                    raise RunError(
                        stage="series",
                        code="FUNC_CALL_FAILED",
                        target=sid,
                        message=f"Error calling function '{func_name}': {e}"
                    )

                if item["obs"] is not None:
                    env[f"{sid}_obs"] = item["obs"]

            else:
                sim = f"{sid}_sim"
                obs = f"{sid}_obs"

                if sim not in env:
                    try:
                        val = read_extract(workPath, simItem)
                        env[sim] = val
                    except Exception as e:
                        raise RunError(
                            stage="series",
                            code="FILE_READ_ERROR",
                            target=sid,
                            message=f"Error reading simulation file for '{sid}': {e}"
                        )

                if obs not in env:
                    if item["obs"] is not None:
                        env[obs] = item["obs"]

        return env