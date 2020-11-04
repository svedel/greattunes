import pytest
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
