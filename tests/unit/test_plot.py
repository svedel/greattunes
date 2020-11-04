import pytest
import torch
from matplotlib.testing.decorators import image_comparison


# @image_comparison(baseline_images=["plot_convergence"], remove_text=True, extensions=['png'])
# def test_plot_plot_convergence(custom_models_simple_training_data_4elements):
#     """
#     does test of convergence plot, leveraging test
#     """
#
#     # train Y data
#     train_Y = custom_models_simple_training_data_4elements[1]
#
#     # define test class
#     class TmpClass:
#         def __init__(self):
#             self.train_Y = train_Y
#
#         from creative_project._plot import plot_convergence
#
#     # initiate class
#     cls = TmpClass()
#
#     # create plot
#     cls.plot_convergence()

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


def test_plot_plot_convergence(custom_models_simple_training_data_4elements):
    """
    does test of convergence plot, leveraging test
    """

    # train Y data
    train_Y = custom_models_simple_training_data_4elements[1]

    # define test class
    class TmpClass:
        def __init__(self):
            self.train_Y = train_Y

        from creative_project._plot import plot_convergence

    # initiate class
    cls = TmpClass()

    # create plot
    fx, ax = cls.plot_convergence()

    # test
    y_axes_res = [-0.3333, 0.7000, 0.7500]
    for it in range(train_Y.shape[0]-1):
        assert ax.lines[0].get_xdata()[it] == it
        assert round(ax.lines[0].get_ydata()[it],4) == round(y_axes_res[it],4)
