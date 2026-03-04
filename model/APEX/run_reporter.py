import queue
import threading
import sqlite3
import csv
import numpy as np
import atexit
import traceback
from pathlib import Path

class RunReporter:
    def __init__(self, backupPath, xLabels, cfg):
        self.cfg = cfg
        self.backupPath = Path(backupPath)
        
        # Clean xLabels to prevent SQL injection or syntax errors
        self.xLabels = [str(x).replace('[','_').replace(']','_').replace('.','_') for x in list(xLabels)]
        
        self.db_path = self.backupPath / "results.db"
        self.summary_csv = self.backupPath / "summary.csv"
        
        self._parse_ids()
        self.start_run_id = 1 
        
        self._q = queue.Queue()
        self._stop = object()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._batch_no = 0
        self._lock = threading.Lock()
        
        atexit.register(self.close)
        
    def _parse_ids(self):
        # Force "_sim" suffix for Series
        self.all_series_ids = [f"{k}_sim" for k in self.cfg.series_index.keys()]
        
        ordered_scalers = []
        seen = set()
        for oid in self.cfg.objectives.use:
            if oid not in seen:
                ordered_scalers.append(oid)
                seen.add(oid)
        for did in self.cfg.diagnostics.use:
            if did not in seen:
                ordered_scalers.append(did)
                seen.add(did)
        for d in self.cfg.derived:
            if d.id not in seen:
                ordered_scalers.append(d.id)
                seen.add(d.id)
        self.all_scaler_ids = ordered_scalers

        rep_cfg = getattr(self.cfg, 'reporter', None)
        if rep_cfg:
            self.out_scaler_ids = getattr(rep_cfg, 'scalars', [])
            self.flush_interval = getattr(rep_cfg, 'flush_interval', 50)
            
            raw_out_series = getattr(rep_cfg, 'series', [])
            self.out_series_ids = []
            for s in raw_out_series:
                if not s.endswith('_sim'):
                    self.out_series_ids.append(f"{s}_sim")
                else:
                    self.out_series_ids.append(s)
        else:
            self.out_series_ids = []
            self.out_scaler_ids = []
            self.flush_interval = 50
    
    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def submit(self, record):
        self._q.put(record)
        
    def new_batch_id(self):
        with self._lock:
            self._batch_no += 1
        return self._batch_no
    
    def _worker(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cols = ["batch_id INTEGER", "run_id INTEGER", "status TEXT", "error TEXT"]
            cols += [f'"{sk}" REAL' for sk in self.all_scaler_ids]
            cols += [f'"{x}" REAL' for x in self.xLabels]
            cursor.execute(f"CREATE TABLE IF NOT EXISTS summary ({', '.join(cols)})")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS series (
                    batch_id INTEGER, run_id INTEGER, series_id TEXT, data BLOB
                )
            """)
            conn.commit()
            
            csv_file = open(self.summary_csv, "a", newline="", encoding="utf-8-sig")
            csv_writer = csv.writer(csv_file)
            
            csv_fields = ["batch_id", "run_id", "status", "error"] + self.out_scaler_ids + self.xLabels
            if csv_file.tell() == 0:
                csv_writer.writerow(csv_fields)
                csv_file.flush()

            series_csv_handlers = {}
            for sk in self.out_series_ids:
                f = open(self.backupPath / f"{sk}.csv", "a", newline="", encoding="utf-8-sig")
                w = csv.writer(f)
                series_csv_handlers[sk] = {'file': f, 'writer': w, 'header_written': f.tell() > 0}
            
            summary_db_buffer = []
            summary_csv_buffer = []
            series_db_buffer = []
            series_csv_buffer = {sk: [] for sk in self.out_series_ids}
            
            holding_pen = {}
            next_expected_id = self.start_run_id 
            current_batch_id = -1 

            while True:
                item = self._q.get()
                
                if item is self._stop:
                    remaining = sorted(holding_pen.keys())
                    self._process_holding_pen(holding_pen, remaining, conn, csv_file, csv_writer,
                                            summary_db_buffer, summary_csv_buffer,
                                            series_db_buffer, series_csv_handlers, series_csv_buffer, 
                                            force_flush=True)
                    self._q.task_done()
                    return
                
                rid = item.get("i", -1) + 1 
                bid = item.get("batch_id", -1)
                
                if bid > current_batch_id:
                    if holding_pen:
                        remaining = sorted(holding_pen.keys())
                        self._process_holding_pen(holding_pen, remaining, conn, csv_file, csv_writer,
                                                summary_db_buffer, summary_csv_buffer,
                                                series_db_buffer, series_csv_handlers, series_csv_buffer, 
                                                force_flush=True)
                    current_batch_id = bid
                    next_expected_id = self.start_run_id 
                    holding_pen.clear()

                holding_pen[rid] = item
                
                ready_ids = []
                while next_expected_id in holding_pen:
                    ready_ids.append(next_expected_id)
                    next_expected_id += 1
                
                if ready_ids:
                    self._process_holding_pen(holding_pen, ready_ids, conn, csv_file, csv_writer,
                                            summary_db_buffer, summary_csv_buffer,
                                            series_db_buffer, series_csv_handlers, series_csv_buffer, 
                                            force_flush=False)
                
                self._q.task_done()
        
        except Exception:
            print("\n🚨 RUN REPORTER CRASHED 🚨")
            traceback.print_exc()
        finally:
            try:
                csv_file.close()
                for h in series_csv_handlers.values():
                    h['file'].close()
                conn.close()
            except:
                pass

    def _process_holding_pen(self, holding_pen, ready_ids, 
                             conn, csv_file, csv_writer,
                             s_db_buf, s_csv_buf, ser_db_buf, ser_csv_h, ser_csv_buf, 
                             force_flush=False):
        
        for rid in ready_ids:
            item = holding_pen.pop(rid)
            
            # Use .item() if these are numpy scalars, otherwise use as is
            batch_id = item.get("batch_id", -1)
            if hasattr(batch_id, 'item'): batch_id = batch_id.item()
            
            run_id = item.get("i", -1)
            # Adjust logical run id (start from 1)
            run_id = run_id + 1 if isinstance(run_id, int) else (run_id.item() + 1 if hasattr(run_id, 'item') else -1)

            status = "error" if "error" in item else "ok"
            err = item.get("error", "")
            
            base_info = [batch_id, run_id, status, err]
            
            # X: .tolist() ensures conversion to native list of floats
            raw_X = item.get("X", [])
            X = np.asarray(raw_X, dtype=float).ravel().tolist()

            # 1. DB Row
            db_row = base_info.copy()
            for key in self.all_scaler_ids:
                val = item.get(key, np.nan)
                # 🌟 [Fix] Pure Native Conversion
                # If it's a Numpy object (array scalar), use .item() to get Python float
                if hasattr(val, 'item'):
                    val = val.item()
                db_row.append(val)
            db_row.extend(X)
            s_db_buf.append(db_row)
            
            # 2. CSV Row
            csv_row = base_info.copy()
            for key in self.out_scaler_ids:
                val = item.get(key, np.nan)
                # 🌟 [Fix] Same here: .item() unwraps numpy scalars cleanly
                if hasattr(val, 'item'):
                    val = val.item()
                csv_row.append(val)
            csv_row.extend(X)
            s_csv_buf.append(csv_row)
            
            # 3. Series Data
            if status == "ok":
                for sk in self.all_series_ids:
                    if sk in item:
                        # Ensure standard numpy float array
                        sim_data = np.asarray(item[sk], dtype=np.float32).ravel()
                        
                        ser_db_buf.append((batch_id, run_id, sk, sim_data.tobytes()))
                        
                        if sk in self.out_series_ids:
                             if sk in ser_csv_buf:
                                 # .tolist() automatically converts elements to python floats
                                 ser_csv_buf[sk].append([batch_id, run_id] + sim_data.tolist())

        if (len(s_db_buf) >= self.flush_interval) or (force_flush and len(s_db_buf) > 0):
            self._flush(conn, csv_file, csv_writer, s_db_buf, s_csv_buf, 
                        ser_db_buf, ser_csv_h, ser_csv_buf)

    def _flush(self, conn, csv_file, csv_writer, summary_db_buffer, summary_csv_buffer, 
               series_db_buffer, series_csv_handlers, series_csv_buffer):
        
        if summary_db_buffer:
            placeholders = ",".join(["?"] * len(summary_db_buffer[0]))
            conn.executemany(f"INSERT INTO summary VALUES ({placeholders})", summary_db_buffer)
            summary_db_buffer.clear()
            
        if summary_csv_buffer:
            csv_writer.writerows(summary_csv_buffer)
            csv_file.flush() 
            summary_csv_buffer.clear()
            
        if series_db_buffer:
            conn.executemany("INSERT INTO series VALUES (?, ?, ?, ?)", series_db_buffer)
            series_db_buffer.clear()

        for sk, rows in series_csv_buffer.items():
            if rows:
                h = series_csv_handlers.get(sk)
                if h:
                    if not h['header_written']:
                        size = len(rows[0]) - 2 
                        header = ["batch_id", "run_id"] + [f"V_{i+1}" for i in range(size)]
                        h['writer'].writerow(header)
                        h['header_written'] = True
                    
                    h['writer'].writerows(rows)
                    h['file'].flush()
                    
                rows.clear()
            
        conn.commit()

    def close(self):
        if not hasattr(self, '_stopped'):
            self._stopped = True
            self._q.put(self._stop)
            if self._thread.is_alive():
                self._thread.join()
