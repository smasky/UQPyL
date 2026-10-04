"""Doe documentation examples.

Migrated from test_review_a07_a15.py; original regression provenance is retained below.
"""

import doctest
import importlib
import pytest


# Regression source: test_review_a07_a15.py::testDoeDocumentationExamplesExecute
@pytest.mark.parametrize("name", ["full_fact", "random", "morris", "saltelli", "fast", "sobol", "lhs"])
def testDoeDocumentationExamplesExecute(name):
    module = importlib.import_module("UQPyL.doe.methods." + name)
    assert doctest.testmod(module).failed == 0
