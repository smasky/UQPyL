from pathlib import Path


def pytest_configure(config):
    # pytest creates basetemp itself, but not missing parent directories.
    # These tests run independently of tests/conftest.py in a fresh CI checkout.
    if config.option.basetemp:
        Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)
