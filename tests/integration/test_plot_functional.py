import pytest
import torch
from botorch.acquisition import ExpectedImprovement
import numpy as np


# test that plot content (what's displayed) works
@pytest.mark.parametrize("sample_method, use_resp_func, num_lines_ax1",
                         [
                             ["manual", False, 2],
                             ["functions", True, 3]
                         ]
                         )
def test_plot_1d_latest_one_window_integration_works(covars_for_custom_models_simple_training_data_4elements,
                                                     covar_details_covars_for_custom_models_simple_training_data_4elements,
                                                     ref_model_and_training_data, sample_method, use_resp_func,
                                                     num_lines_ax1):
    """
    test that plot_1d_latest works and displays the right data on the axes. Test with and without response function
    plotted.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input from fixture
    covars = covars_for_custom_models_simple_training_data_4elements
    covar_details = covar_details_covars_for_custom_models_simple_training_data_4elements[0]
    train_X = ref_model_and_training_data[0]
    train_Y = ref_model_and_training_data[1]
    model_obj = ref_model_and_training_data[2]
    lh = ref_model_and_training_data[3]
    ll = ref_model_and_training_data[4]

    # set response func
    resp_func = None
    colname = list(covar_details.keys())[0]
    if use_resp_func:
        def resp_func(x):
            #return -(6 * x - 2) ** 2 * torch.sin(12 * x - 4)
            z = x[colname].iloc[-1]
            return -(6 * z - 2) ** 2 * np.sin(12 * z - 4)

    # picking out covars
    guesses = [[g[0] for g in covars]]
    lower_bounds = [g[1] for g in covars]
    upper_bounds = [g[2] for g in covars]

    # define test class
    class TmpClass:
        def __init__(self):
            self.initial_guess = torch.tensor(guesses, device=device, dtype=torch.double)
            self.covar_bounds = torch.tensor([lower_bounds, upper_bounds], device=device, dtype=torch.double)
            self.train_X = train_X
            self.proposed_X = train_X
            self.train_Y = train_Y
            self.model = {
                "model_type": "SingleTaskGP",
                "model": model_obj,
                "likelihood": lh,
                "loglikelihood": ll,
                "covars_sampled_iter": train_X.shape[0]
            }
            self.sampling = {
                "method": sample_method, # "functions"
                "response_func": resp_func
            }
            self.covar_details = covar_details

        from greattunes._plot import predictive_results, plot_1d_latest, _covars_ref_plot_1d

    # initialize class
    cls = TmpClass()

    # adding acquisition function to class
    cls.acq_func = {
        "type": "EI",  # define the type of acquisition function
        "object": ExpectedImprovement(model=cls.model["model"], best_f=train_Y.max().item())
    }

    # run the test (test data in both plot axes; parametrize and also test WITH known sampling function)
    ax1, ax2 = cls.plot_1d_latest()

    error_lim = 1e-6

    # ax1 (three lines: observations, mean model, response model; only 2 lines if response model not present)
    # the observations (corresponding to train_X, train_Y)
    assert abs(round(ax1.lines[0].get_xydata()[2, 1], 6) - train_Y[2].item()) < error_lim
    assert abs(round(ax1.lines[0].get_xydata()[0, 0], 6) - train_X[0].item()) < error_lim

    # mean model
    assert ax1.lines[1].get_xydata().shape[0] == 100 # assert that torch.linspace returns a tensor of 100 elements
    assert abs(round(ax1.lines[1].get_xydata()[5, 0], 6) - -1.977778) < error_lim
    assert abs(round(ax1.lines[1].get_xydata()[97, 1], 6) - 0.126714) < error_lim

    # test whether response function is available in plot
    assert len(ax1.lines) == num_lines_ax1

    # test response function values from plot (if available)
    if resp_func is not None:
        assert ax1.lines[2].get_xydata().shape[0] == 100 # assert that torch.linspace returns a tensor of 100 elements
        assert abs(round(ax1.lines[2].get_xydata()[78, 0], 6) - 1.266667) < error_lim
        assert abs(round(ax1.lines[2].get_xydata()[10, 1], 6) - -10.371735) < error_lim

    # ax2 (two lines: acquisition function and latest value)
    assert len(ax2.lines) ==2
    assert ax2.lines[0].get_xydata().shape[0] == 100 # assert that torch.linspace returns a tensor of 100 elements
    assert ax2.lines[1].get_xydata().shape[0] == 1  # assert that only a single point is highlighted (the second "line")

    assert abs(round(ax2.lines[0].get_xydata()[25,0], 6) - -1.088889) < error_lim
    assert abs(round(ax2.lines[0].get_xydata()[60, 1], 6) - 0.00502) < error_lim

    assert abs(round(ax2.lines[1].get_xydata()[0, 0], 6) - 1.0) < error_lim
    assert abs(round(ax2.lines[1].get_xydata()[0, 1], 6) - 0.005185) < error_lim