import atexit
import csv
import queue
import sqlite3
import threading
import traceback
from pathlib import Path

import numpy as np


class RunReporter:
    def __init__(self, backupPath, xLabels, cfg):
        self.cfg = cfg
        self.backupPath = Path(backupPath)
        self.backupPath.mkdir(parents=True, exist_ok=True)

        self.xLabels = self._sanitize_labels(xLabels)

        self.db_path = self.backupPath / "results.db"
        self.summary_csv = self.backupPath / "summary.csv"

        self._parse_ids()
        self.start_run_id = 1

        self._q = queue.Queue()
        self._stop = object()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._batch_no = 0
        self._lock = threading.Lock()

        self._stopped = False
        self._started = False

        atexit.register(self.close)

    def _sanitize_labels(self, labels):
        cleaned = []
        seen = set()

        for i, x in enumerate(labels):
            name = str(x)
            name = name.replace("[", "_").replace("]", "_").replace(".", "_").strip()
            if not name:
                name = f"x_{i+1}"

            base = name
            suffix = 1
            while name in seen:
                suffix += 1
                name = f"{base}_{suffix}"

            seen.add(name)
            cleaned.append(name)

        return cleaned

    def _parse_ids(self):
        self.all_series_ids = [f"{k}_sim" for k in self.cfg.series_index.keys()]

        ordered_scalars = []
        seen = set()

        for oid in self.cfg.objectives.use:
            if oid not in seen:
                ordered_scalars.append(oid)
                seen.add(oid)

        for did in self.cfg.diagnostics.use:
            if did not in seen:
                ordered_scalars.append(did)
                seen.add(did)

        for d in self.cfg.derived:
            if d.id not in seen:
                ordered_scalars.append(d.id)
                seen.add(d.id)

        self.all_scaler_ids = ordered_scalars

        rep_cfg = getattr(self.cfg, "reporter", None)
        if rep_cfg:
            self.out_scaler_ids = list(getattr(rep_cfg, "scalars", []))
            self.flush_interval = int(getattr(rep_cfg, "flush_interval", 50))

            raw_out_series = list(getattr(rep_cfg, "series", []))
            self.out_series_ids = []
            for s in raw_out_series:
                s = str(s)
                if s.endswith("_sim"):
                    self.out_series_ids.append(s)
                else:
                    self.out_series_ids.append(f"{s}_sim")
        else:
            self.out_series_ids = []
            self.out_scaler_ids = []
            self.flush_interval = 50

        if self.flush_interval <= 0:
            self.flush_interval = 50

    def start(self):
        if self._stopped:
            raise RuntimeError("RunReporter is already closed and cannot be restarted.")

        if not self._started:
            self._thread.start()
            self._started = True

    def submit(self, record):
        if self._stopped:
            raise RuntimeError("RunReporter is closed; cannot submit new records.")

        if not self._started:
            self.start()

        self._q.put(record)

    def new_batch_id(self):
        with self._lock:
            self._batch_no += 1
            return self._batch_no

    def _to_scalar_or_nan(self, value):
        if value is None:
            return np.nan

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.size == 0:
                return np.nan
            if arr.size == 1:
                return arr.reshape(-1)[0].item() if hasattr(arr.reshape(-1)[0], "item") else arr.reshape(-1)[0]
            raise ValueError(f"Expected scalar-compatible value, but got shape {arr.shape}")

        return value

    def _to_1d_float_list(self, value):
        if value is None:
            return []

        arr = np.asarray(value, dtype=float).ravel()
        return arr.tolist()

    def _worker(self):
        conn = None
        csv_file = None
        series_csv_handlers = {}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA temp_store=MEMORY")

            cols = ["batch_id INTEGER", "run_id INTEGER", "status TEXT", "error TEXT"]
            cols += [f'"{sk}" REAL' for sk in self.all_scaler_ids]
            cols += [f'"{x}" REAL' for x in self.xLabels]
            cursor.execute(f"CREATE TABLE IF NOT EXISTS summary ({', '.join(cols)})")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS series (
                    batch_id INTEGER,
                    run_id INTEGER,
                    series_id TEXT,
                    data BLOB
                )
                """
            )
            conn.commit()

            csv_file = open(self.summary_csv, "a", newline="", encoding="utf-8-sig")
            csv_writer = csv.writer(csv_file)

            csv_fields = ["batch_id", "run_id", "status", "error"] + self.out_scaler_ids + self.xLabels
            if csv_file.tell() == 0:
                csv_writer.writerow(csv_fields)
                csv_file.flush()

            for sk in self.out_series_ids:
                f = open(self.backupPath / f"{sk}.csv", "a", newline="", encoding="utf-8-sig")
                w = csv.writer(f)
                series_csv_handlers[sk] = {
                    "file": f,
                    "writer": w,
                    "header_written": f.tell() > 0,
                }

            summary_db_buffer = []
            summary_csv_buffer = []
            series_db_buffer = []
            series_csv_buffer = {sk: [] for sk in self.out_series_ids}

            holding_pen = {}
            next_expected_id = self.start_run_id
            current_batch_id = -1

            while True:
                item = self._q.get()

                try:
                    if item is self._stop:
                        remaining = sorted(holding_pen.keys())
                        self._process_holding_pen(
                            holding_pen=holding_pen,
                            ready_ids=remaining,
                            conn=conn,
                            csv_file=csv_file,
                            csv_writer=csv_writer,
                            s_db_buf=summary_db_buffer,
                            s_csv_buf=summary_csv_buffer,
                            ser_db_buf=series_db_buffer,
                            ser_csv_h=series_csv_handlers,
                            ser_csv_buf=series_csv_buffer,
                            force_flush=True,
                        )
                        return

                    rid = item.get("i", -1)
                    rid = rid.item() if hasattr(rid, "item") else rid
                    rid = int(rid) + 1

                    bid = item.get("batch_id", -1)
                    bid = bid.item() if hasattr(bid, "item") else bid
                    bid = int(bid)

                    if bid > current_batch_id:
                        if holding_pen:
                            remaining = sorted(holding_pen.keys())
                            self._process_holding_pen(
                                holding_pen=holding_pen,
                                ready_ids=remaining,
                                conn=conn,
                                csv_file=csv_file,
                                csv_writer=csv_writer,
                                s_db_buf=summary_db_buffer,
                                s_csv_buf=summary_csv_buffer,
                                ser_db_buf=series_db_buffer,
                                ser_csv_h=series_csv_handlers,
                                ser_csv_buf=series_csv_buffer,
                                force_flush=True,
                            )
                        current_batch_id = bid
                        next_expected_id = self.start_run_id
                        holding_pen.clear()

                    holding_pen[rid] = item

                    ready_ids = []
                    while next_expected_id in holding_pen:
                        ready_ids.append(next_expected_id)
                        next_expected_id += 1

                    if ready_ids:
                        self._process_holding_pen(
                            holding_pen=holding_pen,
                            ready_ids=ready_ids,
                            conn=conn,
                            csv_file=csv_file,
                            csv_writer=csv_writer,
                            s_db_buf=summary_db_buffer,
                            s_csv_buf=summary_csv_buffer,
                            ser_db_buf=series_db_buffer,
                            ser_csv_h=series_csv_handlers,
                            ser_csv_buf=series_csv_buffer,
                            force_flush=False,
                        )
                finally:
                    self._q.task_done()

        except Exception:
            print("\nRUN REPORTER CRASHED")
            traceback.print_exc()
        finally:
            try:
                if csv_file is not None:
                    csv_file.close()
            except Exception:
                pass

            for h in series_csv_handlers.values():
                try:
                    h["file"].close()
                except Exception:
                    pass

            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    def _process_holding_pen(
        self,
        holding_pen,
        ready_ids,
        conn,
        csv_file,
        csv_writer,
        s_db_buf,
        s_csv_buf,
        ser_db_buf,
        ser_csv_h,
        ser_csv_buf,
        force_flush=False,
    ):
        for rid in ready_ids:
            item = holding_pen.pop(rid)

            batch_id = item.get("batch_id", -1)
            if hasattr(batch_id, "item"):
                batch_id = batch_id.item()
            batch_id = int(batch_id)

            run_id = item.get("i", -1)
            if hasattr(run_id, "item"):
                run_id = run_id.item()
            run_id = int(run_id) + 1

            status = "error" if "error" in item else "ok"
            err = item.get("error", "")
            err = "" if err is None else str(err)

            base_info = [batch_id, run_id, status, err]

            raw_X = item.get("X", [])
            X = self._to_1d_float_list(raw_X)

            db_row = base_info.copy()
            for key in self.all_scaler_ids:
                try:
                    val = self._to_scalar_or_nan(item.get(key, np.nan))
                except Exception:
                    val = np.nan
                db_row.append(val)
            db_row.extend(X)
            s_db_buf.append(db_row)

            csv_row = base_info.copy()
            for key in self.out_scaler_ids:
                try:
                    val = self._to_scalar_or_nan(item.get(key, np.nan))
                except Exception:
                    val = np.nan
                csv_row.append(val)
            csv_row.extend(X)
            s_csv_buf.append(csv_row)

            if status == "ok":
                for sk in self.all_series_ids:
                    if sk in item:
                        sim_data = np.asarray(item[sk], dtype=np.float32).ravel()
                        ser_db_buf.append((batch_id, run_id, sk, sqlite3.Binary(sim_data.tobytes())))

                        if sk in self.out_series_ids and sk in ser_csv_buf:
                            ser_csv_buf[sk].append([batch_id, run_id] + sim_data.tolist())

        if (len(s_db_buf) >= self.flush_interval) or (force_flush and len(s_db_buf) > 0):
            self._flush(
                conn=conn,
                csv_file=csv_file,
                csv_writer=csv_writer,
                summary_db_buffer=s_db_buf,
                summary_csv_buffer=s_csv_buf,
                series_db_buffer=ser_db_buf,
                series_csv_handlers=ser_csv_h,
                series_csv_buffer=ser_csv_buf,
            )

    def _flush(
        self,
        conn,
        csv_file,
        csv_writer,
        summary_db_buffer,
        summary_csv_buffer,
        series_db_buffer,
        series_csv_handlers,
        series_csv_buffer,
    ):
        if summary_db_buffer:
            placeholders = ",".join(["?"] * len(summary_db_buffer[0]))
            conn.executemany(
                f"INSERT INTO summary VALUES ({placeholders})",
                summary_db_buffer,
            )
            summary_db_buffer.clear()

        if summary_csv_buffer:
            csv_writer.writerows(summary_csv_buffer)
            csv_file.flush()
            summary_csv_buffer.clear()

        if series_db_buffer:
            conn.executemany(
                "INSERT INTO series VALUES (?, ?, ?, ?)",
                series_db_buffer,
            )
            series_db_buffer.clear()

        for sk, rows in series_csv_buffer.items():
            if rows:
                h = series_csv_handlers.get(sk)
                if h:
                    if not h["header_written"]:
                        size = len(rows[0]) - 2
                        header = ["batch_id", "run_id"] + [f"V_{i+1}" for i in range(size)]
                        h["writer"].writerow(header)
                        h["header_written"] = True

                    h["writer"].writerows(rows)
                    h["file"].flush()

                rows.clear()

        conn.commit()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if self._stopped:
            return

        self._stopped = True

        if self._started:
            self._q.put(self._stop)
            self._thread.join()