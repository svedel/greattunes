import torch
import pytest
import numpy as np
from creative_project import CreativeProject


@pytest.mark.parametrize(
    "max_iter, max_response, error_lim, model_type",
    [
        [10, 4.81856, 5e-2, "SingleTaskGP"],
        [50, 6.02073, 1e-3, "SingleTaskGP"],
        [50, 5.99716, 7e-3, "Custom"],
    ]
)
def test_sample_problems_auto_1d_maximization(max_iter, max_response, error_lim, model_type, capsys):
    """
    solve a sample problem in two different conditions.
    test that auto method works for a particular single-covariate (univariate) function
    """

    # define data
    x_input = [(0.5, 0,
                1)]  # covariates come as a list of tuples (one per covariate: (<initial_guess>, <min>, <max>))

    # define response function
    def f(x):
        return -(6 * x["covar0"].iloc[0] - 2) ** 2 * np.sin(12 * x["covar0"].iloc[0] - 4)

    # initialize class instance
    cc = CreativeProject(covars=x_input, model=model_type)

    # run the auto-method
    cc.auto(response_samp_func=f, max_iter=max_iter)

    # assert
    assert cc.model["covars_sampled_iter"] == max_iter

    # assert that max value found
    THEORETICAL_MAX_COVAR = 0.75725
    assert abs(cc.covars_best_response_value[-1].item() - THEORETICAL_MAX_COVAR) < error_lim

    # run current_best method
    cc.current_best()
    captured = capsys.readouterr()

    assert abs(cc.best["covars"][0] - THEORETICAL_MAX_COVAR) < error_lim
    assert abs(cc.best["response"] - max_response) < error_lim
    assert cc.best["iteration_when_recorded"] == max_iter


# 1d maximization problem with rel_tol stopping iteration
@pytest.mark.parametrize(
    "max_iter, rel_tol, rel_tol_steps, num_iterations_exp, response_error_lim, covar_error_lim, model_type",
    [
        [50, 1e-12, None, 2, 1.5, 0.5, "SingleTaskGP"],  # test that iteration stops if relative improvement in one step is below rel_tol
        [50, 1e-12, None, 2, 1.5, 0.5, "Custom"],  # test that iteration stops if relative improvement in one step is below rel_tol
        [50, 1e-10, 5, 8, 2e-1, 0.07, "SingleTaskGP"],  # same as second case above but with realistic rel_tol
        [50, 1e-10, 5, 12, 9e-2, 0.04, "Custom"],  # same as second case above but with realistic rel_tol
    ]
)
def test_sample_problems_auto_1d_maximization_rel_tol_test(max_iter, rel_tol, rel_tol_steps, num_iterations_exp,
                                                           response_error_lim, covar_error_lim, model_type):
    """
    test that 'rel_tol' and 'rel_tol_steps' stops the iteration before max_iter. Test both with rel_tol alone
    (rel_tol_steps = None) as well as with both applied. Using a 1d problem with a known maximum value, the method
    is tested for more than one model type.

    For all cases tests that the number of required iterations is as specified by 'num_iterations_exp'. The goal is
    twofold: 1) that the number of iterations is below max_iter, and 2) that it is the same as 'num_iterations_exp'.

    NOTE: for the first two tests (those with 'rel_tol_steps' = None), the error on the found maximum response is big
    (response_error_lim have been set above 1 to satisfy). We need to continuously monitor this poor convergence
    """

    # define data
    x_input = [(0.5, 0, 1)]

    # define response function
    def f(x):
        return -(6 * x["covar0"].iloc[0] - 2) ** 2 * np.sin(12 * x["covar0"].iloc[0] - 4)

    # initialize class instance
    cc = CreativeProject(covars=x_input, model=model_type)

    # run the auto-method
    cc.auto(response_samp_func=f, max_iter=max_iter, rel_tol=rel_tol, rel_tol_steps=rel_tol_steps)

    # assert that fewer iterations required than max_iter
    assert cc.best_response_value.size()[0] == num_iterations_exp

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

    # assert that max value found (both the covariate and the response)
    THEORETICAL_MAX_COVAR = 0.75725
    MAX_RESPONSE = 6.02073
    assert abs(cc.covars_best_response_value[-1].item() - THEORETICAL_MAX_COVAR) / THEORETICAL_MAX_COVAR  < covar_error_lim
    assert abs(cc.best_response_value[-1].item() - MAX_RESPONSE) / MAX_RESPONSE < response_error_lim


# add new test with multivariate covariates
@pytest.mark.parametrize(
    "max_iter, error_lim, x0_0, x1_0",
    [
        [10, 1.2e-1, 1, -1],
        [50, 5e-3, 1, -1],
        [100, 3e-3, 4, -4],
    ]
)
def test_sample_problems_auto_2d_maximization(max_iter, error_lim, x0_0, x1_0):
    """
    solves a 2D maximzation problem for the negative Easom standard function for optimization. This function has a
    narrow peak at (x_0,x_1) = (0,0), otherwise is flat 0 (see details here https://www.sfu.ca/~ssurjano/easom.html)
    """

    # define the function (negative of the Easom function)
    def f2(x):
        return np.cos(x["covar0"].iloc[0]) * np.cos(x["covar1"].iloc[0]) * np.exp(-(x["covar0"].iloc[0] ** 2 + x["covar1"].iloc[0] ** 2))

    # define the range of interest
    covars2d = [(x0_0, -5, 5.0), (x1_0, -5, 5.0)]

    # initialize class instance
    cc2 = CreativeProject(covars=covars2d)

    # run the auto-method
    cc2.auto(response_samp_func=f2, max_iter=max_iter)

    # run current_best method
    cc2.current_best()

    #error_lim = 5e-3
    x_true = [0, 0]
    y_true = 1

    for it in range(len(covars2d)):
        assert abs(cc2.best["covars"][it] - x_true[it]) < error_lim
    assert abs(cc2.best["response"] - y_true) < error_lim
    assert cc2.best["iteration_when_recorded"] == max_iter