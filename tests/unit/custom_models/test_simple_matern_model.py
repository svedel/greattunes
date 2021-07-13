from botorch.fit import fit_gpytorch_model
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import pytest
from greattunes.custom_models.simple_matern_model import SimpleCustomMaternGP

def test_SimpleCustomMaternGP_definitions(custom_models_simple_training_data,
                                          custom_models_simple_training_data_4elements_covar_details):
    """
    tests that the right models are used, and tests values of Matérn parameter nu
    :param custom_models_simple_training_data: tuple of two input training data (each in torch.tensor format)
    """

    # data
    train_X = custom_models_simple_training_data[0]
    train_Y = custom_models_simple_training_data[1]

    # covar mapping details
    GP_kernel_mapping_covar_identification = custom_models_simple_training_data_4elements_covar_details[2]

    # define the model
    model = SimpleCustomMaternGP(train_X, train_Y, None, GP_kernel_mapping_covar_identification)

    ZERO_TOL = 1e-6

    # assert non-trained model defined with constant mean of 0
    assert "mean_module.constant" in model.state_dict()
    assert abs(model.state_dict()["mean_module.constant"].item()) < ZERO_TOL

    # assert covars module
    assert abs(model.state_dict()["covar_module.raw_outputscale"].item()) < ZERO_TOL
    assert abs(model.state_dict()["covar_module.base_kernel.raw_lengthscale"].item()) < ZERO_TOL


@pytest.mark.parametrize("nu", [None, 2.5, 1.5])
def test_SimpleCustomMaternGP_train(custom_models_simple_training_data,
                                    custom_models_simple_training_data_4elements_covar_details, nu):
    """
    test that the model updates correctly during training
    :param custom_models_simple_training_data: tuple of two input training data (each in torch.tensor format)
    """

    # data
    train_X = custom_models_simple_training_data[0]
    train_Y = custom_models_simple_training_data[1]

    # covar mapping details
    GP_kernel_mapping_covar_identification = custom_models_simple_training_data_4elements_covar_details[2]

    ZERO_TOL = 1e-6

    # define the model
    model = SimpleCustomMaternGP(train_X, train_Y, nu, GP_kernel_mapping_covar_identification)

    # fit the model to the data
    # likelihood
    lh = GaussianLikelihood()

    # define the "loss" function
    ll = ExactMarginalLogLikelihood(lh, model)

    # fit the underlying model
    fit_gpytorch_model(ll)

    # assert that mean has been updated
    assert abs(model.state_dict()["mean_module.constant"].item() - train_Y.item()) < ZERO_TOL

    # assert that covariance has been updated
    assert abs(model.state_dict()["covar_module.raw_outputscale"].item() - -26.510910603516837) < ZERO_TOL


@pytest.mark.parametrize(
    "nu, mean_const, covar_raw_outputscale, covar_raw_lenghtscale",
    [
        [None, 1.410078151215117, 2.5620788497306233, 3.474197969189694],
        [2.5, 1.410078151215117, 2.5620788497306233, 3.474197969189694],
        [1.5, 1.2549585734844588, 2.482677418030433, 4.427168437270844],
    ]
)
def test_SimpleCustomMaternGP_train_4elements(custom_models_simple_training_data_4elements,
                                              custom_models_simple_training_data_4elements_covar_details,
                                              nu, mean_const, covar_raw_outputscale, covar_raw_lenghtscale):
    """
    test that the model updates correctly during training
    :param custom_models_simple_training_data_4elements: tuple of two input training data (each in torch.tensor format)
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # covar mapping details
    GP_kernel_mapping_covar_identification = custom_models_simple_training_data_4elements_covar_details[2]

    ZERO_TOL = 1e-6

    # define the model
    model = SimpleCustomMaternGP(train_X, train_Y, nu, GP_kernel_mapping_covar_identification)

    # fit the model to the data
    # likelihood
    lh = GaussianLikelihood()

    # define the "loss" function
    ll = ExactMarginalLogLikelihood(lh, model)

    # fit the underlying model
    fit_gpytorch_model(ll)

    # assert that mean has been updated
    assert abs(model.state_dict()["mean_module.constant"].item() - mean_const) < ZERO_TOL

    # assert that covariance has been updated
    assert abs(model.state_dict()["covar_module.raw_outputscale"].item() - covar_raw_outputscale) < ZERO_TOL
    assert abs(model.state_dict()["covar_module.base_kernel.raw_lengthscale"].item() - covar_raw_lenghtscale) < ZERO_TOL