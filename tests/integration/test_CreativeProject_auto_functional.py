import pytest
import torch
from creative_project import CreativeProject


@pytest.mark.parametrize(
    "max_iter, max_response, error_lim, model_type",
    [
        [10, 4.81856, 5e-2, "SingleTaskGP"],
        [30, 6.02073, 1e-3, "SingleTaskGP"],
        [10, 5.49629, 5e-2, "Custom"],
        [50, 6.02073, 2.5e-2, "Custom"],
    ]
)
def test_CreativeProject_auto_univariate_functional(max_iter, max_response, error_lim, model_type):
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
    cc = CreativeProject(covars=x_input, model=model_type)

    # run the auto-method
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
    THEORETICAL_MAX_COVAR = 0.75725
    assert abs(cc.covars_best_response_value[-1].item() - THEORETICAL_MAX_COVAR) < error_lim
    assert abs(cc.best_response_value[-1].item() - max_response) < error_lim


@pytest.mark.parametrize(
    "max_iter, max_response, error_lim, model_type",
    [
        [10, 250, 1.1, "SingleTaskGP"],
        [50, 250, 25e-2, "SingleTaskGP"],
        [10, 250, 96e-1, "Custom"],
    ]
)
def test_CreativeProject_auto_multivariate_functional(max_iter, max_response, error_lim, model_type):
    """
    test that auto method works for a particular multivariate (bivariate) function
    """

    # define data
    covars = [(0.5, 0, 1), (0.5, 0, 1)]  # covariates come as a list of tuples (one per covariate: (<initial_guess>, <min>, <max>))

    # define response function
    def f(x):
        return (-(6 * x[0] - 2) ** 2 * torch.sin(12 * x[0] - 4))*(-(6 * x[1] - 2) ** 2 * torch.sin(12 * x[1] - 4))

    # initialize class instance
    cc = CreativeProject(covars=covars, model=model_type)

    # run the auto-method
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
    THEORETICAL_MAX_COVAR = 1.0
    for it in range(len(covars)):
        assert abs(cc.covars_best_response_value[-1, it].item() - THEORETICAL_MAX_COVAR)/THEORETICAL_MAX_COVAR \
               < error_lim
    assert abs(cc.best_response_value[-1].item() - max_response)/max_response < error_lim


# test also printed stuff
@pytest.mark.parametrize("max_iter, max_resp, covar_max_resp",
                         [
                             [2, "-9.09297e-01", "5.00000e-01"],
                             [10, "4.81834e+00", "8.02452e-01"]
                         ])
def test_CreativeProject_auto_printed_to_prompt(max_iter, max_resp, covar_max_resp, capsys):
    """
    tests the stuff printed to the prompt, testing for univariate case
    """

    # define data
    x_input = [(0.5, 0, 1)]

    # define response function
    def f(x):
        return -(6 * x - 2) ** 2 * torch.sin(12 * x - 4)

    # initialize class instance
    cc = CreativeProject(covars=x_input)

    # run the auto-method
    cc.auto(response_samp_func=f, max_iter=max_iter)

    # run current_best method
    cc.current_best()

    captured = capsys.readouterr()

    outtext = ""
    for it in range(1,max_iter+1):
        outtext += "ITERATION " + str(it) + ": Identify new covariate datapoint... Get response for new datapoint... ITERATION  " + str(it) + " - Successfully retrained GP model... Finish iteration...\n"
    outtext += "Maximum response value Y (iteration " + str(it) + "): max_Y =" + max_resp + "\n"
    outtext += "Corresponding covariate values resulting in max_Y: [" + covar_max_resp + "]\n"

    assert captured.out == outtext