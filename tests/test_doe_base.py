from UQPyL.doe.base import Sampler


class DummySampler(Sampler):
    """
    Minimal sampler used only to execute the placeholder logic in `doe/base.py`
    to improve coverage.
    """


def test_sampler_base_init_and_generate_placeholder():
    s = DummySampler()
    # base implementation is a placeholder (pass) and returns None
    assert s._generate(nt=1, nx=1) is None


