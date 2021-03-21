import torch
import pytest
from creative_project import CreativeProject


@pytest.mark.parametrize(
    "max_iter, max_response, error_lim, model_type",
    [
        [10, 4.81856, 5e-2, "SingleTaskGP"],
        [50, 6.02073, 1e-3, "SingleTaskGP"],
        [50, 5.99716, 7e-3, "Custom"],
    ]
)
def test_sample_problems_asktell_1d_maximization(max_iter, max_response, error_lim, model_type, capsys):
    """
    solve a sample problem in two different conditions using the iterative ask-tell approach.
    test that auto method works for a particular single-covariate (univariate) function
    """

    # define data
    x_input = [(0.5, 0,
                1)]  # covariates come as a list of tuples (one per covariate: (<initial_guess>, <min>, <max>))

    # define response function
    def f(x):
        return -(6 * x - 2) ** 2 * torch.sin(12 * x - 4)

    # initialize class instance
    cc = CreativeProject(covars=x_input, model=model_type)

    # run the solution
    for i in range(max_iter):

        # generate candidate
        cc.ask()

        # sample response
        covars = torch.tensor([[cc.proposed_X[-1].item()]], dtype=torch.double)
        response = f(covars)

        # report response
        cc.tell(covars=covars, response=response)


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


# add new test with multivariate covariates
@pytest.mark.parametrize(
    "max_iter, error_lim, x0_0, x1_0",
    [
        [10, 1.2e-1, 1, -1],
        [50, 5e-3, 1, -1],
        [100, 3e-3, 4, -4],
    ]
)
def test_sample_problems_asktell_2d_maximization(max_iter, error_lim, x0_0, x1_0):
    """
    solves a 2D maximzation problem for the negative Easom standard function for optimization. This function has a
    narrow peak at (x_0,x_1) = (0,0), otherwise is flat 0 (see details here https://www.sfu.ca/~ssurjano/easom.html)
    """

    # define the function (negative of the Easom function)
    def f2(x):
        return torch.cos(x[0]) * torch.cos(x[1]) * torch.exp(-(x[0] ** 2 + x[1] ** 2))

    # define the range of interest
    covars2d = [(x0_0, -5, 5.0), (x1_0, -5, 5.0)]

    # initialize class instance
    cc2 = CreativeProject(covars=covars2d)

    # run the auto-method
    #cc2.auto(response_samp_func=f2, max_iter=max_iter)

    # run the solution
    for i in range(max_iter):
        # generate candidate
        cc2.ask()

        # sample response
        covars = torch.tensor([[it.item() for it in cc2.proposed_X[-1]]], dtype=torch.double)
        response = torch.tensor([[f2(cc2.proposed_X[-1]).item()]], dtype=torch.double)

        # report response
        cc2.tell(covars=covars, response=response)

    # run current_best method
    cc2.current_best()

    x_true = [0, 0]
    y_true = 1

    for it in range(len(covars2d)):
        assert abs(cc2.best["covars"][it] - x_true[it]) < error_lim
    assert abs(cc2.best["response"] - y_true) < error_lim
    assert cc2.best["iteration_when_recorded"] == max_iter