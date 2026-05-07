from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RunSession:
    run_id: str
    db_path: str | None = None
    conn: Any = None
    root_dir: str | None = None
    reporter: Any = None
    log_lines: list[str] | None = None

    @property
    def runId(self) -> str:
        return self.run_id

    @property
    def dbPath(self) -> str | None:
        return self.db_path
