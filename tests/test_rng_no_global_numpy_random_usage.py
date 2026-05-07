from pathlib import Path


def test_optimization_and_inference_do_not_use_global_numpy_random_calls():
    root = Path(__file__).resolve().parents[1] / "UQPyL"
    targets = [root / "optimization", root / "inference"]
    allowed = {
        "np.random.default_rng(",
    }

    violations = []
    for base in targets:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "np.random." not in line:
                    continue
                if any(token in line for token in allowed):
                    continue
                violations.append(f"{path.relative_to(root.parent)}:{lineno}: {line.strip()}")

    assert not violations, "Found global numpy random calls:\n" + "\n".join(violations)
