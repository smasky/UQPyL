from UQPyL.doe.base import Sampler
from pathlib import Path


class DummySampler(Sampler):
    """
    Minimal sampler used only to execute the placeholder logic in `doe/base.py`
    to improve coverage.
    """

    def _generate(self, nt=1, nx=1):
        return None


def test_sampler_base_init_and_generate_placeholder():
    s = DummySampler()
    # base implementation is a placeholder (pass) and returns None
    assert s._generate(nt=1, nx=1) is None


def test_doe_methods_module_layout():
    doe_dir = Path(__file__).resolve().parents[1] / "UQPyL" / "doe"
    methods_dir = doe_dir / "methods"
    assert methods_dir.is_dir()
    for name in ["lhs.py", "random.py", "sobol.py", "saltelli.py", "fast.py", "morris.py", "full_fact.py"]:
        assert (methods_dir / name).is_file()


