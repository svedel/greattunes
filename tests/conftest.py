from botorch.models import SingleTaskGP
import numpy as np
import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

### Fixing state of random number generators for test reproducibility
@pytest.fixture(autouse=True)
def rng_state_tests():
    torch.manual_seed(0)
    np.random.seed(0)


### Simple training data
@pytest.fixture(scope="class")
def custom_models_simple_training_data():
    """
    defines very simple dataset for training of custom GP models. Defined in torch.tensor format
    :return: train_X (torch.tensor)
    :return: train_Y (torch.tensor)
    """
    train_X = torch.tensor([[-1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.2]], dtype=torch.double)
    return train_X, train_Y


@pytest.fixture(scope="class")
def custom_models_simple_training_data_4elements():
    """
    defines very simple dataset for training of custom GP models. Defined in torch.tensor format
    :return: train_X (torch.tensor)
    :return: train_Y (torch.tensor)
    """
    train_X = torch.tensor([[-1.0], [-1.1], [-0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.2], [0.15], [0.5], [2.0]], dtype=torch.double)
    return train_X, train_Y


@pytest.fixture(scope="class")
def covars_initialization_data():
    """
    defines simple and more complex initial covariate datasets to test initialization method
    (._initializers.Initializers__initialize_from_covars)
    :return: covar_simple, covar_complex (lists of tuples of doubles)
    """

    covar_simple = [(0.5, 0, 1)]
    covar_complex = [(0.5, 0, 1), (12.5, 8, 15), (-2, -4, 1.1)]
    return covar_simple, covar_complex

@pytest.fixture(scope="class")
def ref_model_and_training_data(custom_models_simple_training_data_4elements):
    """
    defines a simple GP model and the data it is defined by
    :return: train_X, train_Y (training data, from custom_models_simple_training_data_4elements above)
    :return: model_obj (model object, SingleTaskGP)
    :return: lh, ll (model likelihood and marginal log-likelihood)
    """

    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # set up the model
    model_obj = SingleTaskGP(train_X, train_Y)

    # the likelihood
    lh = model_obj.likelihood

    # define the "loss" function
    ll = ExactMarginalLogLikelihood(lh, model_obj)

    return train_X, train_Y, model_obj, lh, ll

    # return model + likelihood
    #self.model["model"] = model_obj
    #self.model["likelihood"] = lh
    #self.model["loglikelihood"] = ll