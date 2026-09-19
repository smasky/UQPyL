from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import config


class BaseReader:
    domain = None

    def __init__(self, dbPath):
        self.dbPath = str(dbPath)
        self.conn = sqlite3.connect(Path(dbPath).resolve().as_uri() + '?mode=rw', uri=True)
        self.conn.row_factory = sqlite3.Row
        try:
            if self.domain is not None:
                row = self.conn.execute("SELECT value FROM runtimeMeta WHERE name='domain'").fetchone()
                if row is None or row[0] != self.domain:
                    raise ValueError(f"Expected a {self.domain} result database.")
        except (sqlite3.DatabaseError, ValueError) as error:
            self.close()
            raise ValueError(f"Expected a {self.domain} result database with runtime metadata.") from error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def get_run(self):
        row = self.conn.execute("SELECT * FROM run LIMIT 1").fetchone()
        return dict(row) if row is not None else None

    def get_run_params(self):
        return {row['name']: row['value'] for row in self.conn.execute('SELECT name, value FROM runParam ORDER BY name')}
    @staticmethod
    def _normalize_result_dir(result_dir) -> Path:
        result_path = Path(result_dir)
        if result_path.is_file():
            result_path = result_path.parent
        result_dir_name = config.resultDirName
        if result_path.name != result_dir_name and (result_path / result_dir_name).exists():
            result_path = result_path / result_dir_name
        elif result_path.name.lower() != "result" and (result_path / "Result").exists():
            result_path = result_path / "Result"
        return result_path

    @classmethod
    def list_runs(cls, result_dir, run_columns: str):
        result_path = cls._normalize_result_dir(result_dir)
        rows = []
        for db_path in sorted(result_path.glob("*.sqlite3")):
            try:
                with cls(db_path) as reader:
                    run = reader.conn.execute(f"SELECT {run_columns} FROM run LIMIT 1").fetchone()
            except (sqlite3.DatabaseError, ValueError):
                run = None
            if run is None:
                continue
            item = dict(run)
            for internalName, publicName in {"runId": "run_id", "createdAt": "created_at",
                                             "finishedAt": "finished_at", "finalFEs": "final_fes",
                                             "finalIters": "final_iters"}.items():
                if internalName in item:
                    item[publicName] = item.pop(internalName)
            item["db_path"] = str(db_path)
            item["file_name"] = db_path.name
            rows.append(item)
        return rows
