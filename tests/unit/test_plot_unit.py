import gpytorch
import pytest
import torch
from botorch.acquisition import ExpectedImprovement
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import pandas as pd
import numpy as np


def test_plot_covars_ref_plot_1d(covars_for_custom_models_simple_training_data_4elements):
    """
    test that correct vector is returned
    """

    # covars
    covars = covars_for_custom_models_simple_training_data_4elements

    lower_bounds = [g[1] for g in covars]
    upper_bounds = [g[2] for g in covars]
    covar_bounds = torch.tensor([lower_bounds, upper_bounds],
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                dtype=torch.double)

    # define test class
    class TmpClass:
        def __init__(self):
            self.covar_bounds = covar_bounds

        from creative_project._plot import _covars_ref_plot_1d

    # initiate class
    cls = TmpClass()

    Xnew, x_min_plot, x_max_plot = cls._covars_ref_plot_1d()

    assert x_max_plot == 2.2
    assert x_min_plot == -2.2
    assert Xnew.shape[0] == 100
    assert round(Xnew[98].item(), 4) == 2.1556
    assert round(Xnew[5].item(), 4) == -1.9778
    assert round(Xnew[20].item(), 4) == -1.3111


def test_plot_covars_ref_plot_1d_fails(covars_initialization_data):
    """
    test that correct vector is returned
    """

    # covars
    covars = covars_initialization_data[1]

    lower_bounds = [g[1] for g in covars]
    upper_bounds = [g[2] for g in covars]
    covar_bounds = torch.tensor([lower_bounds, upper_bounds],
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                dtype=torch.double)

    # define test class
    class TmpClass:
        def __init__(self):
            self.covar_bounds = covar_bounds

        from creative_project._plot import _covars_ref_plot_1d

    # initiate class
    cls = TmpClass()

    with pytest.raises(Exception) as e:
        Xnew, x_min_plot, x_max_plot = cls._covars_ref_plot_1d()
    assert str(e.value) == "kre8_core.creative_project._plot._covars_ref_plot_1d: only valid for 1d data (single covariate), but provided data has 3 covariates."


def test_plot_convergence(custom_models_simple_training_data_4elements):
    """
    does test of convergence plot, leveraging test
    """

    # train Y data
    train_Y = custom_models_simple_training_data_4elements[1]

    # define test class
    class TmpClass:
        def __init__(self):
            self.train_Y = train_Y
            self.best_response_value = train_Y

        from creative_project._plot import plot_convergence

    # initiate class
    cls = TmpClass()

    # create plot
    fx, ax = cls.plot_convergence()

    # test
    y_axes_res = [-0.3333, 0.7000, 0.7500]
    for it in range(train_Y.shape[0]-1):
        assert ax.lines[0].get_xdata()[it] == it
        assert round(ax.lines[0].get_ydata()[it],4) == round(y_axes_res[it], 4)


def test_predictive_results_unit(ref_model_and_training_data):
    """
    test predictive_results for passing tests, using univariate data
    """

    # input from fixture
    train_X = ref_model_and_training_data[0]
    train_Y = ref_model_and_training_data[1]
    model_obj = ref_model_and_training_data[2]
    lh = ref_model_and_training_data[3]
    ll = ref_model_and_training_data[4]

    # define test class
    class TmpClass:
        def __init__(self):
            self.train_X = train_X
            self.train_Y = train_Y
            self.model = {
                "model_type": "SingleTaskGP",
                "model": model_obj,
                "likelihood": lh,
                "loglikelihood": ll,
            }

        from creative_project._plot import predictive_results

    # initialize class
    cls = TmpClass()

    # data at which to retrieve results
    pred_X = torch.linspace(-1.5, 1.5, dtype=torch.double)

    # execute method
    mean_result, lower_bound, upper_bound = cls.predictive_results(pred_X=pred_X)

    error_lim = 1e-6

    # test mean_result
    assert abs(round(mean_result[10].item(), 6) - 0.101979) < error_lim
    assert abs(round(mean_result[75].item(), 6) - 0.490944) < error_lim

    # test lower bound
    assert abs(round(lower_bound[20].item(), 6) - -2.94204) < error_lim
    assert abs(round(lower_bound[66].item(), 6) - -2.828493) < error_lim

    # test upper bound
    assert abs(round(upper_bound[17].item(), 6) - 3.215015) < error_lim
    assert abs(round(upper_bound[95].item(), 6) - 3.626709) < error_lim


def test_plot_best_objective(custom_models_simple_training_data_4elements):
    """
    test that plot_best_objective works if self.train_Y contains 1d data in torch.double data type
    """

    # train Y data
    train_Y = custom_models_simple_training_data_4elements[1]

    # define test class
    class TmpClass:
        def __init__(self):
            self.train_Y = train_Y
            self.best_response_value = train_Y

        from creative_project._plot import plot_best_objective

    # initiate class
    cls = TmpClass()

    # create plot
    fx, ax = cls.plot_best_objective()

    # test
    for it in range(train_Y.shape[0]):
        assert round(ax.lines[0].get_ydata()[it], 4) == round(train_Y[it].item(), 4)


def test_plot_best_objective_fails():
    """
    test that plot_best_objective fails if self.train_Y is None
    """

    # define test class
    class TmpClass:
        def __init__(self):
            self.train_Y = None

        from creative_project._plot import plot_best_objective

    # initiate class
    cls = TmpClass()

    # see that it fails
    with pytest.raises(Exception) as e:
        fx, ax = cls.plot_best_objective()
    assert str(e.value) == "kre8_core.creative_project._plot.plot_best_objective: No objective data: self.train_Y is None"


# test that plot content (what's displayed) works
@pytest.mark.parametrize("sample_method, use_resp_func, num_lines_ax1",
                         [
                             ["manual", False, 2],
                             ["functions", True, 3]
                         ]
                         )
def test_plot_1d_latest_one_window_works(ref_model_and_training_data,
                                         covar_details_covars_for_custom_models_simple_training_data_4elements,
                                         sample_method, use_resp_func, num_lines_ax1,
                                         monkeypatch):
    """
    test that plot_1d_latest works and displays the right data on the axes. Test with and without response function
    plotted. Monkeypatching embedded function calls in plot_1d_latest
    """

    # input from fixture
    train_X = ref_model_and_training_data[0]
    train_Y = ref_model_and_training_data[1]
    model_obj = ref_model_and_training_data[2]
    lh = ref_model_and_training_data[3]
    ll = ref_model_and_training_data[4]
    covar_details = covar_details_covars_for_custom_models_simple_training_data_4elements[0]

    # set response func XXX
    resp_func = None
    colname = list(covar_details.keys())[0]
    if use_resp_func:
        def resp_func(x):
            z = x[colname].iloc[-1]
            return -(6 * z - 2) ** 2 * np.sin(12 * z - 4)
            #return -(6 * x - 2) ** 2 * torch.sin(12 * x - 4)

    # define test class
    class TmpClass:
        def __init__(self):
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

        from creative_project._plot import predictive_results, plot_1d_latest, _covars_ref_plot_1d

    # initialize class
    cls = TmpClass()

    # adding acquisition function to class
    cls.acq_func = {
        "type": "EI",  # define the type of acquisition function
        "object": ExpectedImprovement(model=cls.model["model"], best_f=train_Y.max().item())
    }

    # monkeypatching
    def mock_covars_ref_plot_1d():
        x_min_plot = -1.21
        x_max_plot = 1.1
        Xnew = torch.linspace(x_min_plot, x_max_plot, dtype=torch.double)
        return Xnew, x_min_plot, x_max_plot
    monkeypatch.setattr(
        cls, "_covars_ref_plot_1d", mock_covars_ref_plot_1d
    )

    def mock_predictive_results(pred_X):
        # set to "eval" to get predictive mode
        lh.eval()
        model_obj.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = lh(model_obj(pred_X))

        # Get upper and lower confidence bounds, mean
        lower_bound, upper_bound = observed_pred.confidence_region()
        mean_result = observed_pred.mean

        return mean_result, lower_bound, upper_bound
    monkeypatch.setattr(
        cls, "predictive_results", mock_predictive_results
    )

    # run the test (test data in both plot axes; parametrize and also test WITH known sampling function)
    ax1, ax2 = cls.plot_1d_latest()

    error_lim = 1e-6

    # ax1 (three lines: observations, mean model, response model; only 2 lines if response model not present)
    # the observations (corresponding to train_X, train_Y)
    assert abs(round(ax1.lines[0].get_xydata()[2, 1], 6) - train_Y[2].item()) < error_lim
    assert abs(round(ax1.lines[0].get_xydata()[0, 0], 6) - train_X[0].item()) < error_lim

    # mean model
    assert ax1.lines[1].get_xydata().shape[0] == 100 # assert that torch.linspace returns a tensor of 100 elements
    assert abs(round(ax1.lines[1].get_xydata()[5, 0], 6) - -1.093333) < error_lim
    assert abs(round(ax1.lines[1].get_xydata()[97, 1], 6) - 0.520317) < error_lim

    # test whether response function is available in plot
    assert len(ax1.lines) == num_lines_ax1

    # test response function values from plot (if available)
    if resp_func is not None:
        assert ax1.lines[2].get_xydata().shape[0] == 100 # assert that torch.linspace returns a tensor of 100 elements
        assert abs(round(ax1.lines[2].get_xydata()[78, 0], 6) - 0.61) < error_lim
        assert abs(round(ax1.lines[2].get_xydata()[10, 1], 6) - -0.743607) < error_lim

    # ax2 (two lines: acquisition function and latest value)
    assert len(ax2.lines) ==2
    assert ax2.lines[0].get_xydata().shape[0] == 100 # assert that torch.linspace returns a tensor of 100 elements
    assert ax2.lines[1].get_xydata().shape[0] == 1  # assert that only a single point is highlighted (the second "line")

    assert abs(round(ax2.lines[0].get_xydata()[25,0], 6) - -0.626667) < error_lim
    assert abs(round(ax2.lines[0].get_xydata()[60, 1], 6) - 0.003851) < error_lim

    assert abs(round(ax2.lines[1].get_xydata()[0, 0], 6) - 1.0) < error_lim
    assert abs(round(ax2.lines[1].get_xydata()[0, 1], 6) - 0.005185) < error_lim


# test that plot windows works
def test_plot_1d_latest_multiple_windows_works(ref_model_and_training_data, monkeypatch):
    """
    test that the axes contain multiple images
    """

    # input from fixture
    train_X = ref_model_and_training_data[0]
    train_Y = ref_model_and_training_data[1]
    model_obj = ref_model_and_training_data[2]
    lh = ref_model_and_training_data[3]
    ll = ref_model_and_training_data[4]

    # setting the grid of subfigs for plotting
    n_x = train_X.shape[0]  # number of observations
    n_y = 2  # number of subplots per observation
    fig = plt.figure(figsize=(12, 30))
    outer_gs = gridspec.GridSpec(n_x, n_y)

    # define test class
    class TmpClass:
        def __init__(self):
            self.train_X = train_X
            self.proposed_X = train_X
            self.train_Y = train_Y
            self.model = {
                "model_type": "SingleTaskGP",
                "model": model_obj,
                "likelihood": lh,
                "loglikelihood": ll,
                "covars_sampled_iter": train_X.shape[0],
                "response_sampled_iter": train_Y.shape[0]
            }
            self.sampling = {
                "method": "manual",  # "functions"
                "response_func": None
            }

        from creative_project._plot import predictive_results, plot_1d_latest, _covars_ref_plot_1d

    # initialize class
    cls = TmpClass()

    # adding acquisition function to class
    cls.acq_func = {
        "type": "EI",  # define the type of acquisition function
        "object": ExpectedImprovement(model=cls.model["model"], best_f=train_Y.max().item())
    }

    # monkeypatching
    def mock_covars_ref_plot_1d():
        x_min_plot = -1.21
        x_max_plot = 1.1
        Xnew = torch.linspace(x_min_plot, x_max_plot, dtype=torch.double)
        return Xnew, x_min_plot, x_max_plot

    monkeypatch.setattr(
        cls, "_covars_ref_plot_1d", mock_covars_ref_plot_1d
    )

    def mock_predictive_results(pred_X):
        # set to "eval" to get predictive mode
        lh.eval()
        model_obj.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = lh(model_obj(pred_X))

        # Get upper and lower confidence bounds, mean
        lower_bound, upper_bound = observed_pred.confidence_region()
        mean_result = observed_pred.mean

        return mean_result, lower_bound, upper_bound

    monkeypatch.setattr(
        cls, "predictive_results", mock_predictive_results
    )

    # loop through the steps already taken in optimization run
    for it in range(cls.model['response_sampled_iter']):
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[it])
        _, _ = cls.plot_1d_latest(with_ylabel=False, gs=gs, iteration=it+1)

    # size
    assert len(fig.get_axes()) == n_x * n_y



# remember to duplicate these at integration tests