import torch
import pytest
from creative_project import CreativeProject


@pytest.mark.parametrize(
    "max_iter, max_response, error_lim",
    [
        [10, 4.81856, 5e-2],
        [50, 6.02073, 1e-3],
    ]
)
def test_CreativeProject_auto_functional(max_iter, max_response, error_lim, capsys):
    """
    test that auto method works for a particular single-covariate (univariate) function
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
    cc.auto(response_samp_func=f, max_iter=max_iter)

    # assert
    assert cc.model["covars_sampled_iter"] == max_iter

    # assert that max value found
    theoretical_max_covar = 0.75725
    assert abs(cc.covars_best_response_value[-1].item() - theoretical_max_covar) < error_lim

    # run current_best method
    cc.current_best()
    captured = capsys.readouterr()

    #print(captured.out)
    #print(len(captured.out))

    # # test print statements -- only for first test
    # if max_iter == 10:
    #     #text_output = "('ITERATION 1: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  1 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.5000]], dtype=torch.float64)\n'\n 'ITERATION 2: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  2 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.5000]], dtype=torch.float64)\n'\n 'ITERATION 3: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  3 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.8024]], dtype=torch.float64)\n'\n 'ITERATION 4: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  4 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.8024]], dtype=torch.float64)\n'\n 'ITERATION 5: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  5 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.8024]], dtype=torch.float64)\n'\n 'ITERATION 6: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  6 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.8024]], dtype=torch.float64)\n'\n 'ITERATION 7: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  7 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.8024]], dtype=torch.float64)\n'\n 'ITERATION 8: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  8 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.8024]], dtype=torch.float64)\n'\n 'ITERATION 9: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  9 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.8024]], dtype=torch.float64)\n'\n 'ITERATION 10: Identify new covariate datapoint... Get response for new '\n 'datapoint... ITERATION  10 - Successfully retrained GP model... Finish '\n 'iteration...\n'\n 'max_X\n'\n 'tensor([[0.8024]], dtype=torch.float64)\n'\n 'Maximum response value Y (iteration 10): max_Y =4.818563709373879\n'\n 'Corresponding covariate values resulting in max_Y: [0.8024479419193281]\n')"
    #
    #     text_out = ""
    #     for it in range(1,max_iter+1):
    #         text_out += "ITERATION " + str(it) + ": Identify new covariate datapoint... Get response for new datapoint... ITERATION  " + str(it) + " - Successfully retrained GP model... Finish iteration...\n"
    #     text_out += "Maximum response value Y (iteration 10): max_Y =4.818563709373879\n"
    #     text_out += "Corresponding covariate values resulting in max_Y: [0.8024479419193281]"
    #
    #
    #     for it in range(len(captured.out)):
    #         assert captured.out == text_out #captured.out[it] == text_output[it]

    assert abs(cc.best["covars"][0] - theoretical_max_covar) < error_lim
    assert abs(cc.best["response"] - max_response) < error_lim
    assert cc.best["iteration_when_recorded"] == max_iter



# add new test with multivariate covariates