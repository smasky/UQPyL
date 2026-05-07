import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    ssr = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    if sst == 0.0:
        return 1.0 if ssr == 0.0 else float("-inf")
    return 1.0 - ssr / sst


def make_dataset(name, seed):
    rng = np.random.default_rng(seed)

    if name == "sine_1d":
        x_train = rng.uniform(-1.0, 1.0, size=(40, 1))
        x_test = np.linspace(-1.0, 1.0, 200).reshape(-1, 1)
        y_train = np.sin(3.0 * np.pi * x_train) + 0.15 * x_train
        y_test = np.sin(3.0 * np.pi * x_test) + 0.15 * x_test
        return x_train, y_train, x_test, y_test

    if name == "peaks_2d":
        x_train = rng.uniform(-1.0, 1.0, size=(80, 2))
        x_test = rng.uniform(-1.0, 1.0, size=(300, 2))
        y_train = (
            np.sin(np.pi * x_train[:, [0]])
            + 0.5 * np.cos(2.0 * np.pi * x_train[:, [1]])
            + 0.25 * x_train[:, [0]] * x_train[:, [1]]
        )
        y_test = (
            np.sin(np.pi * x_test[:, [0]])
            + 0.5 * np.cos(2.0 * np.pi * x_test[:, [1]])
            + 0.25 * x_test[:, [0]] * x_test[:, [1]]
        )
        return x_train, y_train, x_test, y_test

    if name == "mixed_3d":
        x_train = rng.uniform(-1.0, 1.0, size=(120, 3))
        x_test = rng.uniform(-1.0, 1.0, size=(400, 3))
        y_train = (
            0.8 * x_train[:, [0]] ** 2
            + np.sin(2.5 * np.pi * x_train[:, [1]])
            + 0.5 * x_train[:, [2]]
            + 0.3 * x_train[:, [0]] * x_train[:, [2]]
        )
        y_test = (
            0.8 * x_test[:, [0]] ** 2
            + np.sin(2.5 * np.pi * x_test[:, [1]])
            + 0.5 * x_test[:, [2]]
            + 0.3 * x_test[:, [0]] * x_test[:, [2]]
        )
        return x_train, y_train, x_test, y_test

    raise ValueError(f"Unknown dataset: {name}")


def safe_import(module_name):
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


def build_models(repo_root):
    sys.path.insert(0, str(repo_root))

    scaler_module, scaler_error = safe_import("UQPyL.util.scaler")
    if scaler_module is None:
        raise RuntimeError(f"Cannot import scaler module: {scaler_error}")

    MinMaxScaler = getattr(scaler_module, "MinMaxScaler")

    def scalers():
        return (MinMaxScaler(0, 1), MinMaxScaler(0, 1))

    specs = [
        ("LinearRegression", "UQPyL.surrogate.regression.linear_regression", "LinearRegression",
         lambda cls: cls(scalers=scalers(), lossType="Origin")),
        ("PolynomialRegression", "UQPyL.surrogate.regression.polynomial_regression", "PolynomialRegression",
         lambda cls: cls(scalers=scalers(), degree=2, lossType="Origin")),
        ("RBF", "UQPyL.surrogate.rbf.radial_basis_function", "RBF",
         lambda cls: cls(scalers=scalers())),
        ("SVR", "UQPyL.surrogate.svr.support_vector_machine", "SVR",
         lambda cls: cls(scalers=scalers(), kernel="rbf")),
        ("MARS", "UQPyL.surrogate.mars.mars", "MARS",
         lambda cls: cls(scalers=scalers(), max_degree=2)),
        ("KRG", "UQPyL.surrogate.kriging.kriging", "KRG",
         lambda cls: cls(scalers=scalers())),
        ("GPR", "UQPyL.surrogate.gp.gaussian_process", "GPR",
         lambda cls: cls(scalers=scalers())),
    ]

    model_factories = {}
    for model_name, module_name, class_name, builder in specs:
        module, error = safe_import(module_name)
        if module is None:
            model_factories[model_name] = {"status": "blocked", "error": error}
            continue

        cls = getattr(module, class_name, None)
        if cls is None:
            model_factories[model_name] = {
                "status": "blocked",
                "error": f"AttributeError: {class_name} not found in {module_name}",
            }
            continue

        model_factories[model_name] = {
            "status": "ok",
            "factory": (lambda klass=cls, make=builder: make(klass)),
        }

    return model_factories


def bench_one_model(model_factory, dataset_names, seeds):
    rows = []

    for dataset_name in dataset_names:
        for seed in seeds:
            x_train, y_train, x_test, y_test = make_dataset(dataset_name, seed)
            model = model_factory()

            fit_start = time.perf_counter()
            model.fit(x_train, y_train)
            fit_sec = time.perf_counter() - fit_start

            pred_start = time.perf_counter()
            train_pred = model.predict(x_train)
            pred_train_sec = time.perf_counter() - pred_start

            test_start = time.perf_counter()
            test_pred = model.predict(x_test)
            pred_test_sec = time.perf_counter() - test_start

            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": int(seed),
                    "train_rmse": rmse(y_train, train_pred),
                    "test_rmse": rmse(y_test, test_pred),
                    "train_r2": r2_score(y_train, train_pred),
                    "test_r2": r2_score(y_test, test_pred),
                    "fit_sec": float(fit_sec),
                    "predict_train_sec": float(pred_train_sec),
                    "predict_test_sec": float(pred_test_sec),
                }
            )

    return rows


def summarize_rows(rows):
    keys = [
        "train_rmse",
        "test_rmse",
        "train_r2",
        "test_r2",
        "fit_sec",
        "predict_train_sec",
        "predict_test_sec",
    ]
    summary = {"n_runs": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Path to the repo root")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--exclude", nargs="*", default=[], help="Model names to exclude")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_names = ["sine_1d", "peaks_2d", "mixed_3d"]
    seeds = [7, 13, 29]

    result = {
        "repo": str(repo_root),
        "datasets": dataset_names,
        "seeds": seeds,
        "models": {},
    }

    model_factories = build_models(repo_root)
    excluded = set(args.exclude)

    for model_name, item in model_factories.items():
        if model_name in excluded:
            result["models"][model_name] = {
                "status": "excluded",
            }
            continue

        if item["status"] != "ok":
            result["models"][model_name] = {
                "status": "blocked",
                "error": item["error"],
            }
            continue

        model_factory = item["factory"]
        try:
            rows = bench_one_model(model_factory, dataset_names, seeds)
            result["models"][model_name] = {
                "status": "ok",
                "summary": summarize_rows(rows),
                "runs": rows,
            }
        except Exception as exc:
            result["models"][model_name] = {
                "status": "error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
