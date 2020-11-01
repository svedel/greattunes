import pytest
import torch
from creative_project import CreativeProject


@pytest.mark.parametrize(
    "max_iter, max_response, error_lim",
    [
        [10, 4.81856, 5e-2],
        [30, 6.02073, 1e-3],
    ]
)
def test_CreativeProject_auto_univariate_functional(max_iter, max_response, error_lim):
    """
    test that auto method works for a particular single-covariate function
    """

    # define data
    x_input = [(0.5, 0,
                1)]  # covariates come as a list of tuples (one per covariate: (<initial_guess>, <min>, <max>))

    # define response function
    def f(x):
        return -(6 * x - 2) ** 2 * torch.sin(12 * x - 4)

    # initialize class instance
    cc = CreativeProject(covars=x_input)

    # run the auto-method
    #max_iter = 10
    cc.auto(response_samp_func=f, max_iter=max_iter)

    # assert that max_iter steps taken by optimizer
    assert cc.model["covars_sampled_iter"] == max_iter
    assert cc.model["covars_proposed_iter"] == max_iter
    assert cc.model["response_sampled_iter"] == max_iter

    # assert that training and test data is stored
    assert cc.train_X.shape[0] == max_iter
    assert cc.proposed_X.shape[0] == max_iter
    assert cc.train_X.shape[0] == max_iter
    assert cc.train_X.shape[1] == 1  # check that it's univariate train_X

    # assert that best response is stored at each step
    assert cc.covars_best_response_value.shape[0] == max_iter
    assert cc.best_response_value.shape[0] == max_iter

    # assert that the correct maximum and covariate values for that spot are identified
    theoretical_max_covar = 0.75725
    assert abs(cc.covars_best_response_value[-1].item() - theoretical_max_covar) < error_lim
    assert abs(cc.best_response_value[-1].item() - max_response) < error_lim


@pytest.mark.parametrize(
    "max_iter, max_response, error_lim",
    [
        [10, 250, 25e-2],
        [20, 250, 3e-3],
    ]
)
def test_CreativeProject_auto_multivariate_functional(max_iter, max_response, error_lim):
    """
    test that auto method works for a particular multivariate (bivariate) function
    """

    #max_iter = 10
    #max_response = 6.02073*6.02073
    #error_lim = 1e-3

    # define data
    covars = [(0.5, 0, 1), (0.5, 0, 1)]  # covariates come as a list of tuples (one per covariate: (<initial_guess>, <min>, <max>))

    # define response function
    def f(x):
        return (-(6 * x[0] - 2) ** 2 * torch.sin(12 * x[0] - 4))*(-(6 * x[1] - 2) ** 2 * torch.sin(12 * x[1] - 4))

    # initialize class instance
    cc = CreativeProject(covars=covars)

    # run the auto-method
    #max_iter = 10
    cc.auto(response_samp_func=f, max_iter=max_iter)

    # assert that max_iter steps taken by optimizer
    assert cc.model["covars_sampled_iter"] == max_iter
    assert cc.model["covars_proposed_iter"] == max_iter
    assert cc.model["response_sampled_iter"] == max_iter

    # assert that training and test data is stored
    assert cc.train_X.shape[0] == max_iter
    assert cc.proposed_X.shape[0] == max_iter
    assert cc.train_X.shape[0] == max_iter
    assert cc.train_X.shape[1] == 2  # check that it's bivariate train_X

    # assert that best response is stored at each step
    assert cc.covars_best_response_value.shape[0] == max_iter
    assert cc.best_response_value.shape[0] == max_iter

    # assert that the correct maximum and covariate values for that spot are identified
    theoretical_max_covar = 1.0
    for it in range(len(covars)):
        assert abs(cc.covars_best_response_value[-1,it].item() - theoretical_max_covar)/theoretical_max_covar < error_lim
    assert abs(cc.best_response_value[-1].item() - max_response)/max_response < error_lim

# test also printed stuff