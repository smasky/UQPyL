import numpy as np
import pytest

from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as GPKernel
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.kriging.kernel import Guass
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.rbf.kernel import Gaussian


@pytest.fixture(params=[(GPR, GPKernel, "l"), (KRG, Guass, "theta"),
                       (RBF, Gaussian, "epsilon")])
def kernelCase(request):
    return request.param


def test_template_reuse_and_training_leave_other_model_unchanged(kernelCase):
    modelClass, kernelClass, parameter = kernelCase
    template = kernelClass()
    original = template.setting.parVal[parameter].copy()
    modelA = modelClass(kernel=template)
    modelB = modelClass(kernel=template)
    assert modelA.kernel is not template
    assert modelB.kernel is not template
    assert modelA.kernel is not modelB.kernel
    assert template.setting.getParaList(tunableOnly=False) == [parameter]

    xTrain = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    modelA.rng = np.random.default_rng(123)
    modelA.fit(xTrain, np.sin(5 * xTrain))
    expected = modelA.predict(xTrain).copy()
    modelB.rng = np.random.default_rng(456)
    modelB.fit(xTrain, np.cos(3 * xTrain))
    np.testing.assert_array_equal(modelA.predict(xTrain), expected)
    np.testing.assert_array_equal(template.setting.parVal[parameter], original)
    assert not hasattr(template, "_templateSetting")


def test_template_edits_only_affect_later_copies_and_setkernel(kernelCase):
    modelClass, kernelClass, parameter = kernelCase
    template = kernelClass()
    modelA = modelClass(kernel=template)
    original = modelA.setting.parVal[parameter].copy()
    template.setting.parVal[parameter][...] = 2.0
    template.setting.parUB[parameter][...] = 7.0
    modelB = modelClass(kernel=template)
    np.testing.assert_array_equal(modelA.setting.parVal[parameter], original)
    np.testing.assert_allclose(modelB.setting.parVal[parameter], 2.0)
    np.testing.assert_allclose(modelB.setting.parUB[parameter], 7.0)

    modelA.setKernel(template)
    assert modelA.kernel is not template
    np.testing.assert_allclose(modelA.setting.parVal[parameter], 2.0)
    template.setting.parVal[parameter][...] = 3.0
    np.testing.assert_allclose(modelA.setting.parVal[parameter], 2.0)
    np.testing.assert_allclose(modelB.setting.parVal[parameter], 2.0)


def test_clone_current_internal_kernel_excludes_model_parameters(kernelCase):
    modelClass, kernelClass, parameter = kernelCase
    model = modelClass(kernel=kernelClass())
    model.setting.parVal[parameter][...] = 2.5
    cloned = model.kernel.clone()
    assert cloned.setting.getParaList(tunableOnly=False) == [parameter]
    np.testing.assert_allclose(cloned.setting.parVal[parameter], 2.5)
    model.setting.parVal[parameter][...] = 3.5
    np.testing.assert_allclose(cloned.setting.parVal[parameter], 2.5)


def test_kernel_choices_snapshot_templates_and_do_not_train_candidates(kernelCase):
    modelClass, kernelClass, parameter = kernelCase
    first, second = kernelClass(), kernelClass()
    second.setting.parVal[parameter][...] = 2.0
    model = modelClass()
    model.setKernelChoices([first, second])
    second.setting.parVal[parameter][...] = 3.0
    model.applyParameterValues(["kernel"], [1.2])
    np.testing.assert_allclose(model.setting.parVal[parameter], 2.0)
    candidate = model.setting.get("kernel")
    assert model.kernel is not candidate
    assert candidate is not second
    assert candidate.setting.getParaList(tunableOnly=False) == [parameter]
    model.setting.parVal[parameter][...] = 4.0
    model.applyParameterValues(["kernel"], [0.2])
    model.applyParameterValues(["kernel"], [1.2])
    np.testing.assert_allclose(model.setting.parVal[parameter], 2.0)
    np.testing.assert_allclose(second.setting.parVal[parameter], 3.0)
