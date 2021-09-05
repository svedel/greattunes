import pandas as pd
import torch
from greattunes.data_format_mappings import (
    tensor2pretty_covariate,
    tensor2pretty_response,
)


@staticmethod
def _find_max_response_value(train_X, train_Y):
    """
    determines best (max) response value max_X across recorded values in train_Y, together with the corresponding
    X values
    :param train_X (torch.tensor)
    :param train_Y (torch.tensor)
    :return max_X (float): the X values corresponding to max_Y
    :return max_Y (float): the maximum Y value recorded
    """

    idmax = train_Y.argmax().item()

    max_X = torch.tensor([train_X[idmax].numpy()], dtype=torch.double)
    max_Y = torch.tensor([train_Y[idmax].numpy()], dtype=torch.double)

    return max_X, max_Y


def _update_max_response_value(self):
    """
    determines the best (max) response across recorded values of train_Y in class instance. Expects that self.train_X,
    self.train_Y exist
    :output
        - self.best_response_value: append latest observation of best Y value
        - self.covars_best_response_value: append with covariates corresponding to best Y value
    """

    try:
        max_X, max_Y = self._find_max_response_value(self.train_X, self.train_Y)
    except Exception:
        raise Exception(
            "greattunes._best_response._update_max_response_value.py: Missing or unable to process "
            "one of following attributes: self.train_X, self.train_Y"
        )

    # the general case: append to existing data structures
    if (
        self.covars_best_response_value is not None
        and self.best_response_value is not None
    ):
        # backend tensor format dataset
        self.covars_best_response_value = torch.cat(
            (self.covars_best_response_value, max_X), dim=0
        )
        self.best_response_value = torch.cat((self.best_response_value, max_Y), dim=0)

        # pretty data format (pandas)
        self.covars_best_response = self.covars_best_response.append(
            tensor2pretty_covariate(
                train_X_sample=max_X, covar_details=self.covar_details
            )
        )
        self.best_response = self.best_response.append(
            tensor2pretty_response(train_Y_sample=max_Y)
        )

    # initializing: set the first elements
    else:

        # backend tensor format
        self.covars_best_response_value = max_X
        self.best_response_value = max_Y

        # pretty data format (pandas)
        self.covars_best_response = tensor2pretty_covariate(
            train_X_sample=max_X, covar_details=self.covar_details
        )
        self.best_response = tensor2pretty_response(train_Y_sample=max_Y)


def current_best(self):
    """
    prints to prompt the latest estimate of best value (max) of response (Y), together with the corresponding
    covariates (X)

    assumes:
        - model, likelihood exists
        - some list of observations exists
    does:
        - returns current best estimate of objective + COVARIATES RESPONSIBLE (MISSING THE COVARIATES BIT)
    """

    # max response Y (float) -- assumes univariate
    max_Y = self.best_response_value[-1].item()

    # corresponding covariates X (list of float)
    # convert to pretty format before printing to prompt
    max_X_df = tensor2pretty_covariate(
        train_X_sample=self.covars_best_response_value[-1].reshape(
            1, self.total_num_covars
        ),
        covar_details=self.covar_details,
    )

    # print to prompt
    print(
        "Maximum response value Y (iteration "
        + str(self.model["response_sampled_iter"])
        + "): max_Y = "
        + "{:.5e}".format(max_Y)
    )
    print(
        "Corresponding covariate values resulting in max_Y:\n\t"
        + max_X_df.to_string(index=False).replace("\n", "\n\t")
    )

    # set attributes
    self.best = {
        "covars": max_X_df,  # max_X_list,
        "response": pd.DataFrame({"Response": [max_Y]}),  # max_Y,
        "iteration_when_recorded": self.model["response_sampled_iter"],
    }


def _update_proposed_data(self, candidate):
    """
    update the record of proposed covariates for next observation ("proposed_X") and associated counter
    "model["covars_proposed_iter"]".
    :param candidate (1 X num_covars torch tensor)
    :return
        - updates
            - self.model["covars_proposed_iter"]
            - self.proposed_X

    assumes:
        - proposed covariates ("proposed_X"), actually recorded covariates ("train_X") and observations
        ("train_Y") must follow each other (i.e. "proposed_X" can only be ahead by one iteration relative to
        "train_X" and "train_Y")
    """

    # data type validation on "candidate"
    assert isinstance(candidate, torch.DoubleTensor), (
        "greattunes._best_response._update_proposed_data: provided variable is not 'candidate' of type "
        "torch.DoubleTensor (type of 'candidate' is " + str(type(candidate)) + ")"
    )

    # check number of covariates in "candidate" (only if previous records exist)
    if self.proposed_X is not None:
        assert (
            candidate.size()[1] == self.proposed_X.size()[1]
        ), "greattunes._best_response._update_proposed_data: wrong number of covariates provided in " "'candidate'. Expected " + str(
            self.proposed_X.size()[1]
        ) + ", but got " + str(
            candidate.size()[1]
        )

    # takes latest sampled covariates and store proposal as next
    # update counter
    self.model["covars_proposed_iter"] = self.model["covars_sampled_iter"] + 1

    # === add proposed candidate ===
    # special case if no data stored previously (in which case self.proposed_X is None)
    if self.proposed_X is None:
        self.proposed_X = candidate
    # special case where overwriting a previous proposal
    elif self.proposed_X.shape[0] == self.model["covars_proposed_iter"]:
        self.proposed_X[-1] = candidate
    else:
        self.proposed_X = torch.cat((self.proposed_X, candidate), dim=0)


def best_predicted(self):
    """
    returns the best predicted value from the surrogate model, assuming best means maximum. Returns the maximum value
    of the mean model, as well as the maximum from the mean model minus the one standard deviation of the noise model
    """

    # generate candidates randomly
    x_init = torch.rand(1, self.covar_bounds.size(1)).requires_grad_(True)
    x_init = (
        self.covar_bounds[0] + (self.covar_bounds[1] - self.covar_bounds[0]) * x_init
    )

    # best of mean model
    ym, xm = self._find_best_predicted(x_init=x_init, type="mean")
    xm_df = tensor2pretty_covariate(train_X_sample=xm, covar_details=self.covar_details)

    # best of lower confidence region
    ylc, xlc = self._find_best_predicted(x_init=x_init, type="lower_conf")
    xlc_df = tensor2pretty_covariate(
        train_X_sample=xlc, covar_details=self.covar_details
    )

    # print to prompt
    print(
        "Best predicted response value Y (mean model): max_Y = "
        + "{:.5e}".format(ym.item())
    )
    print(
        "Corresponding covariate values resulting in max_Y:\n\t"
        + xm_df.to_string(index=False).replace("\n", "\n\t")
        + "\n"
    )
    print(
        "Best predicted response value Y (lower confidence region): max_Y = "
        + "{:.5e}".format(ylc.item())
    )
    print(
        "Corresponding covariate values resulting in max_Y:\n\t"
        + xlc_df.to_string(index=False).replace("\n", "\n\t")
    )


def _evaluate_model(self, type, x):
    """
    evaluate model defined by self.model["model"] at point 'x' based on input 'type'

    :param type (str): paramter used to determined whether to maximize the mean ("mean") or the lower confidence region
    ("lower_conf")
    :param x (torch tensor): point to evaluate function at

    Depends on
        - self.model["model"]: must be a function operating on a tensor that has the attributes "mean" and
        "confidence_region". BoTorch Gaussian process models have the characteristics

    :return y (tensor): response from function in self.model["model"] at point 'x' evaluated according to 'type'
    """

    tmp = self.model["model"](x)
    if type == "mean":
        y = tmp.mean
    elif type == "lower_conf":
        y = tmp.confidence_region()[0]

    return y


def _find_best_predicted(
    self, x_init, type, maximize=True, max_iter=500, simplex_std_lim=1e-3,
):
    """
    apply the Nelder-Mead simplex method to find maximum of the Gaussian process model in self.model["model"].

    :param x_init (torch tensor, size 1 X num_covars): initial guess to start evaluation series to find maximum value. To be done in space of
    tensor variables (behind-the-scenes continuous variables)
    :param type (str): paramter used to determined whether to maximize the mean ("mean") or the lower confidence region
    ("lower_conf")
    :param max_iter (int, default=500): maximum iteration number, used as stopping criterion
    :param simplex_std_lim (float, default=0.5): maximum variation (standard deviation) allow for simplex points, used
    as stopping criterion
    :return curr_max (tensor, doubletype, size 1 X 1): optimum function value
    :return c (tensor, doubletype, size num_covars X 1): covariate datapoint for curr_max
    """

    factor = 1.0
    if not maximize:
        factor = -1.0

    # create initial simplex from initial datapoint x_init. simplex is a tensor, size num_covars+1 X num_covars with the
    # collection of points for search, one row is one point
    simplex = x_init
    n_covar = x_init.size()[1]
    for i in range(n_covar):  # iterate through number of covariate dimensions
        # unit vector in direction i
        ui = torch.zeros(1, n_covar, dtype=self.dtype, device=self.device)
        ui[0, i] = 1

        # factor
        h = 0.05
        if x_init[0, i] == 0:
            h = 0.00025
        simplex = torch.cat((simplex, x_init + h * ui), 0)

    # parameters
    alpha = 1  # for reflection
    gamma = 2  # for expansion
    beta = 0.5  # for contraction
    delta = 0.6  # for shrink contraction

    # initialize
    prev_max = None
    curr_max = torch.rand(1, 1, dtype=self.dtype, device=self.device)
    iter = 0
    iter_cont = True

    while prev_max is None or iter_cont:

        # === update prev_max and iteration counter ===
        prev_max = curr_max
        iter += 1

        # === next iteration of response function ===
        # evaluate function at simplex points
        y = factor * self._evaluate_model(type, simplex)

        # sort the points from smallest to largest based on function values
        vals, ids = torch.sort(y.squeeze())

        # rearrange points in simplex based on response value
        simplex = simplex[ids]

        # compute the centroid over all points but the worst
        c = simplex[1:, :].mean(dim=0).unsqueeze(0)
        curr_max = factor * self._evaluate_model(type, c)

        # errors
        # rel_err = torch.abs((curr_max - prev_max) / curr_max)
        # abs_err = torch.abs(curr_max - prev_max)

        # update variation in simplex points (used for stopping)
        simplex_std = torch.std(simplex)

        # === update simplex ===
        # reflection
        xr = c + alpha * (c - simplex[0])
        val_xr = factor * self._evaluate_model(type, xr)

        # decision
        # reflection is betweem second worst and best in simplex: replace worst point
        if vals[1].item() < val_xr.item() and val_xr.item() <= vals[-1].item():
            vals[0] = val_xr
            simplex[0] = xr

        # reflection is worse than second worst point in simplex: contract to avoid continuing moving in wrong direction
        elif val_xr.item() < vals[1].item():
            xc = c + beta * (simplex[0] - c)
            val_xc = factor * self._evaluate_model(type, xc)

            # contraction point better than current worst point
            if val_xc.item() > vals[0].item():
                vals[0] = val_xc
                simplex[0] = xc

            # contraction point is worse than current worst point: shrink contraction to redefine whole simplex except
            # best point
            else:
                # update simplex
                simplex = simplex[-1] + delta * (simplex - simplex[-1])

        # reflection is better than any point in simplex: expand to hope to find better, replace worst point
        # (vals[-1].item() < val_xr.item():)
        else:
            xe = c + gamma * (xr - c)
            val_xe = factor * self._evaluate_model(type, xe)

            # expanded point gives bigger response
            if val_xe.item() >= val_xr.item():
                vals[0] = val_xe
                simplex[0] = xe

            # reflection point gives bigger response
            elif val_xe.item() < val_xr.item():
                vals[0] = val_xr
                simplex[0] = xr

        # if (rel_err <= rel_tol or abs_err <= abs_tol or iter >= max_iter or simplex_std <= simplex_std_lim):
        if iter >= max_iter or simplex_std <= simplex_std_lim:
            if iter > 1:
                iter_cont = False

    return curr_max, c
