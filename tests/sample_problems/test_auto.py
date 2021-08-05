import torch
import pytest
import numpy as np
from greattunes import TuneSession
from scipy.stats import multivariate_normal


@pytest.mark.parametrize(
    "max_iter, max_response, error_lim, model_type",
    [
        [10, 4.81856, 5e-2, "SingleTaskGP"],
        [50, 6.02073, 1e-3, "SingleTaskGP"],
        [50, 5.99716, 9e-3, "Custom"],
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
    cc = TuneSession(covars=x_input, model=model_type)

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

    assert abs(cc.best["covars"].values[0][0] - THEORETICAL_MAX_COVAR) < error_lim
    assert abs(cc.best["response"].values[0][0] - max_response) < error_lim
    assert cc.best["iteration_when_recorded"] == max_iter


# 1d maximization problem with rel_tol stopping iteration
@pytest.mark.parametrize(
    "max_iter, rel_tol, rel_tol_steps, num_iterations_exp, response_error_lim, covar_error_lim, model_type",
    [
        [50, 1e-12, None, 2, 1.5, 0.5, "SingleTaskGP"],  # test that iteration stops if relative improvement in one step is below rel_tol
        [50, 1e-12, None, 2, 1.5, 0.5, "Custom"],  # test that iteration stops if relative improvement in one step is below rel_tol
        [50, 1e-10, 5, 8, 2e-1, 0.07, "SingleTaskGP"],  # same as second case above but with realistic rel_tol
        [50, 1e-10, 5, 15, 9e-2, 0.04, "Custom"],  # same as second case above but with realistic rel_tol
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
    cc = TuneSession(covars=x_input, model=model_type)

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
    cc2 = TuneSession(covars=covars2d)

    # run the auto-method
    cc2.auto(response_samp_func=f2, max_iter=max_iter)

    # run current_best method
    cc2.current_best()

    #error_lim = 5e-3
    x_true = [0, 0]
    y_true = 1

    for it in range(len(covars2d)):
        assert abs(cc2.best["covars"].values[0][it] - x_true[it]) < error_lim
    assert abs(cc2.best["response"].values[0][0] - y_true) < error_lim
    assert cc2.best["iteration_when_recorded"] == max_iter


def test_full_multicategorical_problem():
    """
    test that the framework works on a full problem where the response depends on both continuous, integer and
    categorical variables.

    In this problem the goal is to find the optimal brownie recipe among the variables
    variables: flour, egg, sugar, chocolate_type, chocolate_amount, butter, nut_type, nut_amount

    Egg is an integer, chocolate_type and nut_type are categorical and the rest are continuous. The objective function
    is modeled as a multivariate Gaussian for all numerical variables, with a separate function for each combination of
    categorical variables
    """

    # === define variables for the objective function ===

    # helper function
    def sym_update(arr, entry, val):
        """
        updates entries in symmetric matrices: updates array 'arr' at positions 'entry'=[i][j] and the mirror [j][i] to
        the new value 'val'
        """
        xi = entry[0]
        yi = entry[1]
        arr[xi][yi] = val
        arr[yi][xi] = val
        return arr

    # === taste multiplication factors ===
    # the rank order
    # 1 white chocolate + almonds (right amounts)
    # 2 dark chocolate + hazelnuts (right amounts)
    # 3 dark chocolate + almonds
    # 4 white chocolate + hazelnuts

    mf_wa = 1.2  # white chocolate <> almonds
    mf_dh = 1.18  # dark chocolate <> hazelnuts
    mf_da = 1.12  # dark chocolate <> almonds
    mf_wh = 0.95  # white chocolate <> hazelnuts

    # === variables ===
    # flour, egg, sugar, chocolate_type, chocolate_amount, butter, nut_type, nut_amount
    # order of numerical variables
    # [flour, egg, sugar, chocolate_amount, butter, nut_amount]

    # === setting covariate structure and defining models ===

    # DARK CHOCOLATE + HAZELNUTS

    # correlations
    rho = np.identity(6)  # initialize

    rho = sym_update(rho, (0, 1), 0.5)  # flour-egg
    rho = sym_update(rho, (0, 2), 0.3)  # flour-sugar
    rho = sym_update(rho, (0, 3), 0.3)  # flour-chocolate_amount
    rho = sym_update(rho, (0, 4), 0.1)  # flour-butter
    rho = sym_update(rho, (0, 5), 0.1)  # flour-nut_amount

    rho = sym_update(rho, (1, 2), 0.5)  # egg-sugar
    rho = sym_update(rho, (1, 3), 0.5)  # egg-chocolate_amount
    rho = sym_update(rho, (1, 4), 0.1)  # egg-butter
    rho = sym_update(rho, (1, 5), 0.1)  # egg-nut_amount

    rho = sym_update(rho, (2, 3), 0.0)  # sugar-chocolate_amount
    rho = sym_update(rho, (2, 4), 0.5)  # sugar-butter
    rho = sym_update(rho, (2, 5), 0.0)  # sugar-nut_amount

    rho = sym_update(rho, (3, 4), 0.5)  # chocolate_amount-butter
    rho = sym_update(rho, (3, 5), 0.3)  # chocolate_amount-nut_amount

    rho = sym_update(rho, (4, 5), 0.0)  # butter-nut_amount

    # variances
    sigv = np.array([[50, 2, 50, 30, 30, 30]])  # standard deviations
    var_arr = np.dot(sigv.T, sigv)

    # covariances
    cv_arr = np.multiply(rho, var_arr)

    # mean
    # [flour, egg, sugar, chocolate_amount, butter, nut_amount]
    mean_arr = np.array([125, 3, 150, 300, 100, 100])

    # the model
    rv_dh = multivariate_normal(mean_arr, cv_arr)
    rv_dh_norm = rv_dh.pdf(mean_arr)

    # DARK CHOCOLATE + ALMONDS

    # correlations
    rho = np.identity(6)  # initialize

    rho = sym_update(rho, (0, 1), 0.5)  # flour-egg
    rho = sym_update(rho, (0, 2), 0.3)  # flour-sugar
    rho = sym_update(rho, (0, 3), 0.3)  # flour-chocolate_amount
    rho = sym_update(rho, (0, 4), 0.1)  # flour-butter
    rho = sym_update(rho, (0, 5), 0.1)  # flour-nut_amount

    rho = sym_update(rho, (1, 2), 0.5)  # egg-sugar
    rho = sym_update(rho, (1, 3), 0.3)  # egg-chocolate_amount
    rho = sym_update(rho, (1, 4), 0.1)  # egg-butter
    rho = sym_update(rho, (1, 5), 0.1)  # egg-nut_amount

    rho = sym_update(rho, (2, 3), 0.0)  # sugar-chocolate_amount
    rho = sym_update(rho, (2, 4), 0.5)  # sugar-butter
    rho = sym_update(rho, (2, 5), 0.0)  # sugar-nut_amount

    rho = sym_update(rho, (3, 4), 0.3)  # chocolate_amount-butter
    rho = sym_update(rho, (3, 5), 0.0)  # chocolate_amount-nut_amount

    rho = sym_update(rho, (4, 5), 0.0)  # butter-nut_amount

    # variances
    sigv = np.array([[50, 2, 50, 80, 30, 30]])  # standard deviations
    var_arr = np.dot(sigv.T, sigv)

    # covariances
    cv_arr = np.multiply(rho, var_arr)

    # mean
    # [flour, egg, sugar, chocolate_amount, butter, nut_amount]
    mean_arr = np.array([125, 3, 150, 300, 100, 100])

    # the model
    rv_da = multivariate_normal(mean_arr, cv_arr)
    rv_da_norm = rv_da.pdf(mean_arr)

    # WHITE CHOCOLATE + ALMONDS

    # correlations
    rho = np.identity(6)  # initialize

    rho = sym_update(rho, (0, 1), 0.5)  # flour-egg
    rho = sym_update(rho, (0, 2), 0.3)  # flour-sugar
    rho = sym_update(rho, (0, 3), 0.2)  # flour-chocolate_amount
    rho = sym_update(rho, (0, 4), 0.1)  # flour-butter
    rho = sym_update(rho, (0, 5), 0.1)  # flour-nut_amount

    rho = sym_update(rho, (1, 2), 0.5)  # egg-sugar
    rho = sym_update(rho, (1, 3), 0.7)  # egg-chocolate_amount
    rho = sym_update(rho, (1, 4), 0.3)  # egg-butter
    rho = sym_update(rho, (1, 5), 0.1)  # egg-nut_amount

    rho = sym_update(rho, (2, 3), 0.4)  # sugar-chocolate_amount
    rho = sym_update(rho, (2, 4), 0.5)  # sugar-butter
    rho = sym_update(rho, (2, 5), 0.0)  # sugar-nut_amount

    rho = sym_update(rho, (3, 4), 0.3)  # chocolate_amount-butter
    rho = sym_update(rho, (3, 5), 0.4)  # chocolate_amount-nut_amount

    rho = sym_update(rho, (4, 5), 0.0)  # butter-nut_amount

    # variances
    sigv = np.array([[25, 2, 40, 50, 30, 20]])  # standard deviations
    var_arr = np.dot(sigv.T, sigv)

    # covariances
    cv_arr = np.multiply(rho, var_arr)

    # mean
    # [flour, egg, sugar, chocolate_amount, butter, nut_amount]
    mean_arr = np.array([125, 3, 150, 300, 100, 100])

    # the model
    rv_wa = multivariate_normal(mean_arr, cv_arr)
    rv_wa_norm = rv_wa.pdf(mean_arr)

    # WHITE CHOCOLATE + HAZELNUTS

    # correlations
    rho = np.identity(6)  # initialize

    rho = sym_update(rho, (0, 1), 0.5)  # flour-egg
    rho = sym_update(rho, (0, 2), 0.3)  # flour-sugar
    rho = sym_update(rho, (0, 3), 0.2)  # flour-chocolate_amount
    rho = sym_update(rho, (0, 4), 0.1)  # flour-butter
    rho = sym_update(rho, (0, 5), 0.1)  # flour-nut_amount

    rho = sym_update(rho, (1, 2), 0.5)  # egg-sugar
    rho = sym_update(rho, (1, 3), 0.7)  # egg-chocolate_amount
    rho = sym_update(rho, (1, 4), 0.3)  # egg-butter
    rho = sym_update(rho, (1, 5), 0.1)  # egg-nut_amount

    rho = sym_update(rho, (2, 3), 0.4)  # sugar-chocolate_amount
    rho = sym_update(rho, (2, 4), 0.5)  # sugar-butter
    rho = sym_update(rho, (2, 5), 0.0)  # sugar-nut_amount

    rho = sym_update(rho, (3, 4), 0.3)  # chocolate_amount-butter
    rho = sym_update(rho, (3, 5), 0.2)  # chocolate_amount-nut_amount

    rho = sym_update(rho, (4, 5), 0.4)  # butter-nut_amount

    # variances
    sigv = np.array([[25, 2, 20, 80, 30, 50]])  # standard deviations
    var_arr = np.dot(sigv.T, sigv)

    # covariances
    cv_arr = np.multiply(rho, var_arr)

    # mean
    # [flour, egg, sugar, chocolate_amount, butter, nut_amount]
    mean_arr = np.array([125, 3, 100, 300, 100, 100])

    # the model
    rv_wh = multivariate_normal(mean_arr, cv_arr)
    rv_wh_norm = rv_wh.pdf(mean_arr)

    # === objective function ===
    def brownie_score(xdf):
        """
        score of the brownie (the objective function).
        :param xdf (pandas df)
        :return score (array or element)
        """

        # === get numerical content ===
        nrows = xdf.shape[0]

        flour = np.reshape(xdf["flour"].values, (nrows, 1))
        egg = np.reshape(xdf["egg"].values, (nrows, 1))
        sugar = np.reshape(xdf["sugar"].values, (nrows, 1))
        chocolate_amount = np.reshape(xdf["chocolate_amount"].values, (nrows, 1))
        butter = np.reshape(xdf["butter"].values, (nrows, 1))
        nut_amount = np.reshape(xdf["nut_amount"].values, (nrows, 1))

        entry = np.hstack((flour, egg, sugar, chocolate_amount, butter, nut_amount))

        # === get categorical entries ===
        chocolate_type = xdf["chocolate_type"].values
        nut_type = xdf["nut_type"].values

        # combines the categorical inputs
        full_type = list(map('-'.join, zip(chocolate_type, nut_type)))

        # === determines score ===
        #output = np.zeros((1, nrows))
        output = np.zeros((nrows))

        for i in range(nrows):
            # picks the right model to use
            if full_type[i] == "dark-almond":
                output[i] = mf_da * rv_da.pdf(entry[i, :])/rv_da_norm  # normalize by largest value so baseline at mean is 1
            elif full_type[i] == "dark-hazelnut":
                output[i] = mf_dh * rv_dh.pdf(entry[i, :])/rv_dh_norm
            elif full_type[i] == "white-almond":
                output[i] = mf_wa * rv_wa.pdf(entry[i, :])/rv_wa_norm
            elif full_type[i] == "white-hazelnut":
                output[i] = mf_wh * rv_wh.pdf(entry[i, :])/rv_wh_norm

        return output

    # === solve the problem ===
    # let's try to solve the problem

    # variables: flour, egg, sugar, chocolate_type, chocolate_amount, butter, nut_type, nut_amount

    # define covariates
    covars_brownie = {
        "flour": {
            "guess": 100,
            "min": 0,
            "max": 200,
            "type": float,
        },
        "egg": {
            "guess": 2,
            "min": 0,
            "max": 5,
            "type": int,
        },
        "sugar": {
            "guess": 100,
            "min": 50,
            "max": 200,
            "type": float,
        },
        "chocolate_type": {
            "guess": "dark",
            "options": {"dark", "white"},
            "type": str,
        },
        "chocolate_amount": {
            "guess": 300,
            "min": 150,
            "max": 500,
            "type": float,
        },
        "butter": {
            "guess": 100,
            "min": 0,
            "max": 200,
            "type": float,
        },
        "nut_type": {
            "guess": "almond",
            "options": {"almond", "hazelnut"},
            "type": str,
        },
        "nut_amount": {
            "guess": 100,
            "min": 0,
            "max": 200,
            "type": float,
        },
    }

    # initialize class instance
    cc = TuneSession(covars=covars_brownie)

    # number of iterations
    max_iter = 100

    # run the auto-method
    cc.auto(response_samp_func=brownie_score, max_iter=max_iter)

    # run current_best method
    cc.current_best()

    # print best responses
    print(cc.best_response)

    # print responses
    print(cc.y_data)

    # === assert execution ===
    assert cc.best_response.shape[0] == max_iter
    assert cc.y_data.shape[0] == max_iter
    assert cc.covars_best_response.shape[0] == max_iter
    assert cc.x_data.shape[0] == max_iter

    # colnames for covariates
    for colname in cc.covars_best_response.columns:
        assert colname in list(covars_brownie.keys())

    # check that all categorical variables only take the accepted values
    for cname in ["chocolate_type", "nut_type"]:
        for j in cc.x_data[cname].values:
            assert j in covars_brownie[cname]["options"]

    # check that integer variable is all integers
    for j in cc.x_data["egg"].values:
        assert isinstance(j, np.int64)
        assert (j >= covars_brownie["egg"]["min"])&(j <= covars_brownie["egg"]["max"])

    # === assert performance ===
    #y_true = 1e-1
    #error_lim = 1e-1
    #assert abs(cc.best["response"].values[0][0] - y_true) < error_lim




