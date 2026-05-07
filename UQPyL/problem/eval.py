from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Eval:
    objs: Optional[np.ndarray] = None
    cons: Optional[np.ndarray] = None
    sim: Optional[np.ndarray] = None

    @property
    def hasObjs(self) -> bool:
        return self.objs is not None

    @property
    def hasCons(self) -> bool:
        return self.cons is not None

    @property
    def hasSim(self) -> bool:
        return self.sim is not None
