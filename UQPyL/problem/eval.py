from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Eval:
    objs: Optional[np.ndarray] = None
    cons: Optional[np.ndarray] = None
    sims: Optional[np.ndarray] = None
    target: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.target not in (None, "objs", "cons", "sims"):
            raise ValueError("The target must be None, 'objs', 'cons' or 'sims'.")

        if self.target == "objs":
            self.cons = None
            self.sims = None
        elif self.target == "cons":
            self.objs = None
            self.sims = None
        elif self.target == "sims":
            self.objs = None
            self.cons = None

    @property
    def hasObjs(self) -> bool:
        return self.objs is not None

    @property
    def hasCons(self) -> bool:
        return self.cons is not None

    @property
    def hasSims(self) -> bool:
        return self.sims is not None
