import matplotlib.pyplot as plt
import copy
import gpytorch
import torch
import pandas as pd


def _covars_ref_plot_1d(self):
    """
    creates reference covars data for plotting surrogate model and acquisition functions across the range
    defined when initiaizing class. Only 1D covars
    """

    # check that only 1d data accepted
    if self.covar_bounds.shape[1] > 1:
        raise Exception(
            "greattunes.greattunes._plot._covars_ref_plot_1d: only valid for 1d data (single "
            "covariate), but provided data has "
            + str(self.covar_bounds.shape[1])
            + " covariates."
        )

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
    :param pred_X (torch.tensor type torch.double): input covariate datapoints (must have dtype=torch.double)
    :return mean_result (torch.tensor)
    :return lower_bound (torch.tensor)
    :return upper_bound (torch.tensor)
    """

    # set model to produce predictive results
    model_local = copy.deepcopy(self.model["model"])  # self.model["model"]
    model_local.eval()
    model_local.likelihood.eval()
    # likelihood_local = copy.deepcopy(
    #     self.model["likelihood"]
    # )  # self.model["likelihood"]
    # likelihood_local.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model_local.likelihood(
            model_local(pred_X)
        )  # likelihood_local(model_local(pred_X))

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
    :param self (object): initiated instance of TuneSession class with optimization having run for at least
        the number of iterations given by iteration
    :param kwargs:
        - gs: object of type matplotlib.gridspec.GridSpecFromSubplotSpec
        - iteration: last iteration number of stored data to plot (still using models from latest available
            iteration in self, which is stored in self.model["covars_sampled_iter"]). Must start at 1, not 0
    :return ax1 (matplotlib axes): first plot of surrogate model and interrogation points (train_X points)
    :return ax2 (matplotlib axes): acquisition function and latest point picked
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
        # Xnew_df = tensor2pretty_covariate(train_X_sample=Xnew, covar_details=self.covar_details)
        # actual_resp = self.sampling["response_func"](Xnew)
        # actual_resp = self.sampling["response_func"](Xnew_df)
        colname = list(self.covar_details.keys())[0]
        actual_resp = [
            self.sampling["response_func"](pd.DataFrame({colname: [x.item()]}))
            for x in Xnew
        ]

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
        ax1.plot(Xnew.numpy(), actual_resp, "--k", label="Actual Response")

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

    return ax1, ax2


def plot_convergence(self):
    """
    plot relative improvement in response variable against iteration number.
    TODO: add "rel_tol" relative tolerance to plot when that is defined
    :input:
        - self.train_Y (torch.tensor of dtype=torch.double): observations (batch_shape X num_obs X num_output_models
            [allows for batched models] OR num_obs X num_output_models)
    :return: fx (figure)
    :return: ax (figure axes)
    """

    # calculates the relative error
    y = copy.deepcopy(self.best_response_value)
    y_diff = y[1:] - y[:-1]
    y_rel = y_diff / y[1:]

    # build the plot
    fx, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(list(range(y_rel.shape[0])), y_rel.numpy(), "-b.")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration $n$")
    ax.set_ylabel("Relative improvement between iterations, $(y_n-y_{n-1})/y_n$")

    return fx, ax


def plot_best_objective(self):
    """
    plots best objective value as a function of iteration number
    :input:
        - self.train_Y (torch.tensor of dtype=torch.double): observations (batch_shape X num_obs X num_output_models
            [allows for batched models] OR num_obs X num_output_models)
    :return: fx (figure)
    :return: ax (figure axes)
    """

    if self.train_Y is None:
        raise Exception(
            "greattunes.greattunes._plot.plot_best_objective: No objective data: self.train_Y "
            "is None"
        )

    # build the plot
    fx, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(list(range(self.train_Y.shape[0])), self.best_response_value.numpy(), "-b.")
    ax.set_xlabel("Iteration $n$")
    ax.set_ylabel("Best objective $y^{max}_n$ found up to iteration $n$")

    return fx, ax


# def plot_GP_samples(self, num_realizations=25):
#     """
#     plot sample traces of individual realizations of the underlying Gaussian Process model stored in
#     instantiated TuneSession-object self
#     :param self (object): initiated instance of TuneSession class with optimization having run for at least
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
