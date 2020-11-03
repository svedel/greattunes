import matplotlib.pyplot as plt
import copy
import gpytorch
import torch


def _covars_ref_plot_1d(self):
    """
    creates reference covars data for plotting surrogate model and acquisition functions across the range
    defined when initiaizing class. Only 1D covars
    """

    # x-data for plotting covariates
    # find the natural scale of the problem as the max absolute number of values in the range. Coverts to float
    Xnew_scale = self.covar_bounds.abs().numpy().max()
    x_min_plot = self.covar_bounds[0].item() - 0.1 * Xnew_scale
    x_max_plot = self.covar_bounds[1].item() + 0.1 * Xnew_scale
    Xnew = torch.linspace(x_min_plot, x_max_plot, dtype=torch.double)

    return Xnew, x_min_plot, x_max_plot


# Model output for plotting
def predictive_results(self, pred_X):
    """
    provides predictive results (mean, std) for the stored, trained model at the covariate datapoints
    provided by pred_X
    NOTE: AM INCLUDING self AS PARAM, SHOULD USE INSTANTIATED CLASS VIA self WHEN WRAPPING IN CLASS
    :param pred_X (torch.tensor type torch.double): input covariate datapoints (must have dtype=torch.double)
    :return mean_result (torch.tensor)
    :return lower_bound (torch.tensor)
    :return upper_bound (torch.tensor)
    """

    # set model to produce predictive results
    model_local = copy.deepcopy(self.model["model"])  # self.model["model"]
    model_local.eval()
    likelihood_local = copy.deepcopy(
        self.model["likelihood"]
    )  # self.model["likelihood"]
    likelihood_local.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood_local(model_local(pred_X))

    # Get upper and lower confidence bounds, mean
    lower_bound, upper_bound = observed_pred.confidence_region()
    mean_result = observed_pred.mean

    return mean_result, lower_bound, upper_bound


def plot_1d_latest(self, with_ylabel=True, **kwargs):
    """
    simple plotting function of surrogate model and acquisition function at the cycle number given by
    iteration. All data up to that point will also be shown, i.e. the acquisition function used to obtain
    latest data point will be shown
    :param iteration (int): iteration number in optimization for which to plot
    :param self (object): initiated instance of CreativeProject class with optimization having run for at least
        the number of iterations given by iteration
    :param kwargs:
        - gs: object of type matplotlib.gridspec.GridSpecFromSubplotSpec
        - iteration: last iteration number of stored data to plot (still using models from latest available
            iteration in self, which is stored in self.model["covars_sampled_iter"]). Must start at 1, not 0
    """

    # use plotting grid if provided as input
    if kwargs.get("gs") is not None:
        gs = kwargs.get("gs")
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

    else:
        fx, ax = plt.subplots(2, 1, figsize=(6, 6))
        ax1 = ax[0]
        ax2 = ax[1]

    # getting data

    # getting iteration
    if kwargs.get("iteration") is not None:
        iteration = int(kwargs.get("iteration"))
    else:
        iteration = self.model["covars_sampled_iter"]
    iteration_idex = iteration - 1

    # covars data for plotting
    Xnew, x_min_plot, x_max_plot = self._covars_ref_plot_1d()

    # Gaussian process results
    mean_result, lower_bound, upper_bound = self.predictive_results(Xnew)

    # get actual response if this is a known function
    include_resp = False
    if (
        self.sampling["method"] == "functions"
        and self.sampling["response_func"] is not None
    ):
        include_resp = True
        actual_resp = self.sampling["response_func"](Xnew)

    # acquisition function
    acq_func = self.acq_func["object"](Xnew.unsqueeze(-1).unsqueeze(-1)).detach()
    acq_point = self.acq_func["object"](
        self.proposed_X[iteration_idex].unsqueeze(-1)
    ).detach()
    acq_covar = self.proposed_X[iteration_idex]

    # plotting surrogate model with data

    # plot training data
    ax1.plot(
        self.train_X[:iteration_idex].numpy(),
        self.train_Y[:iteration_idex].numpy(),
        "k*",
        label="Observed Data",
    )

    # Plot predictive means
    ax1.plot(Xnew.numpy(), mean_result.numpy(), "b", label="Mean")

    # Add actual response (if known)
    if include_resp:
        ax1.plot(Xnew.numpy(), actual_resp.numpy(), "--k", label="Actual Response")

    # Shade between the lower and upper confidence bounds
    ax1.fill_between(
        Xnew.numpy(),
        lower_bound.numpy(),
        upper_bound.numpy(),
        alpha=0.5,
        label="Confidence",
    )

    # set title, legend and y label
    ax1.set_title("Iteration " + str(iteration))
    ax1.legend()
    if with_ylabel:
        ax1.set_ylabel("Gaussian process regression")

    # plotting acquisition function

    # plot acquisition function
    ax2.plot(Xnew.numpy(), acq_func.numpy())

    # selected point
    ax2.plot(
        acq_covar.numpy(),
        acq_point.numpy(),
        "^",
        markersize=8,
        label="x" + str(iteration) + ": " + str(acq_covar.item()),
    )

    ax2.legend()
    if with_ylabel:
        ax2.set_ylabel("Acquisition function (" + self.acq_func["type"] + ")")


# def plot_GP_samples(self, num_realizations=25):
#     """
#     plot sample traces of individual realizations of the underlying Gaussian Process model stored in
#     instantiated CreativeProject-object self
#     :param self (object): initiated instance of CreativeProject class with optimization having run for at least
#         the number of iterations given by iteration
#     :num_realizations (int): number of samples to plot
#     """
#
#     f, ax = plt.subplots(1, 1, figsize=(6, 6))
#
#     # covars data for plotting
#     Xnew, x_min_plot, x_max_plot = covars_ref_plot_1d(self)
#     expanded_Xnew = Xnew.unsqueeze(0).repeat(num_samples, 1, 1)  # for plotting realizations
#
#     # Gaussian process results
#     mean_result, lower_bound, upper_bound = predictive_results(Xnew, self)
