from __future__ import annotations

import sqlite3
from pathlib import Path


class BaseReader:
    @staticmethod
    def _normalize_result_dir(result_dir) -> Path:
        result_path = Path(result_dir)
        if result_path.is_file():
            result_path = result_path.parent
        if result_path.name.lower() != "result" and (result_path / "Result").exists():
            result_path = result_path / "Result"
        return result_path

    @classmethod
    def list_runs(cls, result_dir, run_columns: str):
        result_path = cls._normalize_result_dir(result_dir)
        rows = []
        for db_path in sorted(result_path.glob("*.sqlite3")):
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            try:
                run = conn.execute(f"SELECT {run_columns} FROM run LIMIT 1").fetchone()
            except sqlite3.OperationalError:
                run = None
            finally:
                conn.close()
            if run is None:
                continue
            item = dict(run)
            if "runId" in item:
                item["run_id"] = item.pop("runId")
            item["dbPath"] = str(db_path)
            item["fileName"] = db_path.name
            rows.append(item)
        return rows
