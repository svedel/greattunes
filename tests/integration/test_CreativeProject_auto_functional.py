import pandas as pd
import pytest
import torch
import numpy as np
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
        return -(6 * x['covar0'].iloc[0] - 2) ** 2 * np.sin(12 * x['covar0'].iloc[0] - 4)

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
        [50, 250, 1, "SingleTaskGP"],
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
        return (-(6 * x['covar0'].iloc[0] - 2) ** 2 * np.sin(12 * x['covar0'].iloc[0] - 4)) * (-(6 * x['covar1'].iloc[0] - 2) ** 2 * np.sin(12 * x['covar1'].iloc[0] - 4))

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


@pytest.mark.parametrize(
    "max_iter, rel_tol, rel_tol_steps, num_iterations_exp",
    [
        [50, 0.01, None, 2],  # test that iteration stops if relative improvement in one step is below rel_tol
        [50, 0.01, 2, 5],  # test that iteration stops if relative improvement in two consecutive steps is below rel_tol
        [50, 0.001, 2, 5],  # test that iteration stops if relative improvement in two consecutive steps is below rel_tol
        [50, 0.001, 4, 7],  # test that iteration stops if relative improvement in three consecutive steps is below rel_tol
        [50, 1e-8, 5, 8],  # same as second case above but with realistic rel_tol
    ]
)
def test_CreativeProject_auto_rel_tol_test(max_iter, rel_tol, rel_tol_steps, num_iterations_exp):
    """
    test that 'rel_tol' and 'rel_tol_steps' stops the iteration before max_iter. Test both with rel_tol alone
    (rel_tol_steps = None) as well as with both applied. Use a simple single-variable problem to run the tests

    For all cases tests that the number of required iterations does not exceed those obtained during testing locally
    """

    # define data
    x_input = [(0.5, 0, 1)]

    # define response function
    def f(x):
        return -(6 * x['covar0'].iloc[0] - 2) ** 2 * np.sin(12 * x['covar0'].iloc[0] - 4)

    # initialize class instance
    cc = CreativeProject(covars=x_input)

    # run the auto-method
    cc.auto(response_samp_func=f, max_iter=max_iter, rel_tol=rel_tol, rel_tol_steps=rel_tol_steps)

    # assert that fewer iterations required than max_iter
    assert cc.best_response_value.size()[0] <= num_iterations_exp

    # define local function to calculate relative improvement
    def cal_rel_improvements(best_response_tensor, rel_tol, rel_tol_steps):

        # special case where rel_tol_steps = None is dealt with by setting rel_tol_steps = 1 (since in this case only
        # look back one step)
        if rel_tol_steps is None:
            rel_tol_steps = 1

        all_below_rel_tol = False

        # first build tensor with the last rel_tol_steps entries in self.best_response_value and the last
        # rel_tol_steps+1 entries
        tmp_array = torch.cat(
            (best_response_tensor[-(rel_tol_steps + 1):-1], best_response_tensor[-(rel_tol_steps):]),
            dim=1).numpy()

        # calculate the relative differences
        tmp_rel_diff = np.diff(tmp_array, axis=1) / best_response_tensor[-(rel_tol_steps):].numpy()

        # determine if all below 'rel_tol'
        below_rel_tol = [rel_dif[0] < rel_tol for rel_dif in tmp_rel_diff.tolist()]

        # only accept if the relative difference is below 'rel_tol' for all steps
        if sum(below_rel_tol) == rel_tol_steps:
            all_below_rel_tol = True

        return all_below_rel_tol

    # assert that relative improvement is below rel_tol for rel_tol_steps number of steps. For special case of
    # rel_tol_steps = None only investigate last step
    assert cal_rel_improvements(cc.best_response_value, rel_tol, rel_tol_steps)


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
        return -(6 * x['covar0'].iloc[0] - 2) ** 2 * np.sin(12 * x['covar0'].iloc[0] - 4)

    # initialize class instance
    cc = CreativeProject(covars=x_input)

    # run the auto-method
    cc.auto(response_samp_func=f, max_iter=max_iter)

    # run current_best method
    cc.current_best()

    captured = capsys.readouterr()

    outtext = ""
    for it in range(1,max_iter+1):
        outtext += "ITERATION " + str(it) + ": Identify new covariate datapoint... Get response for new datapoint... Successfully trained GP model... Finish iteration...\n"
    outtext += "Maximum response value Y (iteration " + str(it) + "): max_Y =" + max_resp + "\n"
    outtext += "Corresponding covariate values resulting in max_Y: [" + covar_max_resp + "]\n"

    assert captured.out == outtext