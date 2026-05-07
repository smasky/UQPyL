import json
import os
import pickle
import re
import uuid
from datetime import datetime

import numpy as np


def slugify_name(name):
    text = str(name).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "problem"


def make_run_id(method_name, problem_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = uuid.uuid4().hex[:4]
    problem_slug = slugify_name(problem_name)
    return f"{str(method_name).lower()}_{problem_slug}_{timestamp}_{suffix}"


def ensure_result_dir(root_dir):
    result_dir = os.path.join(root_dir, "Result")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def build_db_path(root_dir, method_name, problem_name):
    result_dir = ensure_result_dir(root_dir)
    run_id = make_run_id(method_name, problem_name)
    return os.path.join(result_dir, f"{run_id}.sqlite3"), run_id


def array_to_json(value):
    if value is None:
        return None
    return json.dumps(np.asarray(value).tolist(), ensure_ascii=True)


def from_json_array(text):
    if text is None:
        return None
    return np.asarray(json.loads(text))


def pickle_to_blob(value):
    return value if value is None else pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def spawn_seed(rng):
    return int(rng.integers(0, 2**32 - 1))


def export_runtime_meta(*, run_id, method, problem_name, n_input, n_output, n_con, runtime, created_at, extra=None):
    payload = {
        "run_id": run_id,
        "method": method,
        "problem_name": problem_name,
        "n_input": n_input,
        "n_output": n_output,
        "n_con": n_con,
        "runtime": runtime,
        "created_at": created_at,
    }
    if extra:
        payload.update(extra)
    return payload


def export_reader_summary(*, run_id, method, problem_name, n_input, n_output, n_con, runtime, created_at, finished_at, extra=None):
    payload = export_runtime_meta(
        run_id=run_id,
        method=method,
        problem_name=problem_name,
        n_input=n_input,
        n_output=n_output,
        n_con=n_con,
        runtime=runtime,
        created_at=created_at,
        extra=extra,
    )
    payload["finished_at"] = finished_at
    return payload
