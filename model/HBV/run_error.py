from dataclasses import dataclass

@dataclass
class RunError(Exception):
    stage: str       # subprocess / series / derived / evaluator / config
    code: str        # TIMEOUT / NONZERO_EXIT / SIZE_MISMATCH / EXPR_EVAL_FAILED ...
    target: str      # simulation / tn / tn_r2 / objective_xxx
    message: str

    def __str__(self):
        return f"{self.stage}:{self.code}:{self.target}: {self.message}"
    
