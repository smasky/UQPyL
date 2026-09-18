import sys
from pathlib import Path


# Make `import UQPyL` work when running tests from repo root without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    # pytest creates basetemp itself, but its parents must already exist.
    if config.option.basetemp:
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)

