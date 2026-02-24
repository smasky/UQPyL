from pathlib import Path
import queue
from concurrent.futures import ThreadPoolExecutor
from extract_handler import expand_row_ranges
import threading
import numpy as np
import csv

class RunReporter:
    def __init__(self, backupPath, xLabels, series_ids, obj_ids, diag_ids, obj_sids_map, diag_sids_map, cfg,
                 sep=",", float_fmt=".5f"):
        
        self.backupPath = Path(backupPath)
        
        self.x_csv = self.backupPath / "X_used.csv"
        self.sum_csv = self.backupPath / "summary.csv"
        
        self.xLabels = list(xLabels)
        self.obj_ids = list(obj_ids)
        self.diag_ids = list(diag_ids)
        self.series_ids = list(series_ids)
        
        # TODO
        self.obj_sids_map = obj_sids_map
        self.diag_sids_map = diag_sids_map
        # 
        
        self.x_fields = ["batch_id", "run_id", "status", "error"] + self.xLabels
        
        sum_fields = ["batch_id", "run_id", "status", "error"] + self.obj_ids + self.diag_ids
        for oid in self.obj_ids:
            for sid in self.obj_sids_map.get(oid, []):
                sum_fields.append(f"{oid}_{sid}")
        for did in self.diag_ids:
            for sid in self.diag_sids_map.get(did, []):
                sum_fields.append(f"{did}_{sid}")
        self.sum_fields = sum_fields
        
        # series
        
        self.series_dir = {}
        for sid in self.series_ids:
            
            file =self.backupPath / f"{sid}.csv"
            row_ranges = cfg.series_index.get(sid, {}).sim.rowRanges
            offsets = expand_row_ranges(row_ranges)
            size = len(offsets)
            fields = ["batch_id", "run_id"] + [f"V_{i+1}" for i in range(size)]
            
            self.series_dir[sid] = {
                "file": file,
                "fields": fields,
            }
        
        self.sep = sep
        self.float_fmt = float_fmt
        
        #
        self._q = queue.Queue()
        self._stop = object()
        self._thread = threading.Thread(target=self._worker, daemon=True)

        self._batch_no = 0
        self._run_no = 0
        self._lock = threading.Lock()
        
    def start(self):
        self._thread.start()
    
    def new_batch_id(self):
        
        with self._lock:
            self._batch_no += 1
            
        return self._batch_no
    
    def _fmt(self, x):
        try:
            return format(float(x), self.float_fmt)
        except Exception:
            return ""
    
    def submit(self, record):
        self._q.put(record)
    
    def close(self):
        self._q.put(self._stop)
        self._q.join()
    
    def _empty_row(self, fields):
        return {k: "" for k in fields}
    
    def _worker(self):
        
        fx = open(self.x_csv, "a", newline="", encoding="utf-8-sig")
        fs = open(self.sum_csv, "a", newline="", encoding="utf-8-sig")
        
        xw = csv.DictWriter(fx, fieldnames=self.x_fields, delimiter=self.sep, extrasaction="ignore")
        sw = csv.DictWriter(fs, fieldnames=self.sum_fields, delimiter=self.sep, extrasaction="ignore")
        
        sid_handlers = {}
        
        def get_sid_handler(sid):
            if sid in sid_handlers:
                return sid_handlers[sid]
            
            file = self.series_dir[sid]["file"]
            fields = self.series_dir[sid]["fields"]
            
            f = open(file, "a", newline="", encoding="utf-8-sig")
            w = csv.DictWriter(f, fieldnames=fields, delimiter=self.sep, extrasaction="ignore")
            
            if f.tell() == 0:
                w.writeheader()
                f.flush()
            sid_handlers[sid] = {'file': f, 'writer': w}
            return {'file': f, 'writer': w}
            
        if fx.tell() == 0:
            xw.writeheader(); fx.flush()
        if fs.tell() == 0:
            sw.writeheader(); fs.flush()
        
        try:
            while True:
                item = self._q.get()
                try:
                    
                    if item is self._stop:
                        return
                    
                    batch_id = item.get("batch_id", "NA")
                    run_id = item.get("i")
                    
                    status = "error" if "error" in item else "ok"
                    err = item.get("error", "")
                    
                    # X_used.csv
                    x_row = self._empty_row(self.x_fields)
                    x_row.update({"batch_id": batch_id, "run_id": run_id, "status": status, "error": err})
                    X = np.asarray(item.get("X", []), dtype=float).ravel()
                    for k, name in enumerate(self.xLabels):
                        if k < len(X):
                            x_row[name] = self._fmt(X[k])
                    xw.writerow(x_row); fx.flush()
                    
                    # summary.csv
                    s_row = self._empty_row(self.sum_fields)
                    s_row.update({"batch_id": batch_id, "run_id": run_id, "status": status, "error": err})
                    
                    if status == 'ok':
                        for oid in self.obj_ids:
                            if oid in item:
                                s_row[oid] = self._fmt(item[oid]['agg_val'])
                                for sid in self.obj_sids_map.get(oid, []):
                                    s_row[f"{oid}_{sid}"] = self._fmt(item[oid]['vals'][sid])        
                        for did in self.diag_ids:
                            if did in item:
                                s_row[did] = self._fmt(item[did]['agg_val'])
                                for sid in self.diag_sids_map.get(did, []):
                                    s_row[f"{did}_{sid}"] = self._fmt(item[did]['vals'][sid])

                    sw.writerow(s_row); fs.flush()
                    
                    if status == 'ok':
                        series_cache = item.get("cache", None)
                        if series_cache:
                            for sid in self.series_ids:
                                fh = get_sid_handler(sid)
                                vv = series_cache.get(sid, None)
                                if not vv or vv.get("sim") is None:
                                    continue
                                
                                sim = np.asarray(vv["sim"], dtype=float).ravel()
                                
                                row = {"batch_id": batch_id, "run_id": run_id}
                                for k in range(1, sim.size + 1):
                                    row[f"V_{k}"] = self._fmt(sim[k-1])
                                
                                fh['writer'].writerow(row); fh['file'].flush()
                finally:
                    self._q.task_done()
                    
        finally:
            fx.close()
            fs.close()
            
            for fh in sid_handlers.values():    
                fh['file'].close()