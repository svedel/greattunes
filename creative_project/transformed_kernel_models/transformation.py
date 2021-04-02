import torch


# kernel transformation mapping
def GP_kernel_transform(x, GP_kernel_mapping_covar_identification):
    """
    performs transformation of covariates to enable Gaussian Process models to also do Bayesian optimization for
    integer and categorical variables. The transformation is part of the solution described in this paper:
    E.C. Garrido-Merchán and D. Hernandéz-Lobato: Dealing with categorical and integer-valued variables in Bayesian
    Optimization with Gaussian processes, Neurocomputing vol. 380, 7 March 2020, pp. 20-35
    (https://arxiv.org/pdf/1805.03463.pdf, https://www.sciencedirect.com/science/article/abs/pii/S0925231219315619)

    Briefly the transformation applies only to the continuous variables, and is applied only inside the GP kernel. The
     transformation does the following:
    * for integers: integers covariates (still handled via a single continuous covariate in the GP), the continuous GP
    variable is mapped to the nearest integer (via rounding)
    * for categorical: categorical covariates are handled via one-hot encoding to continuous variables in the GP; the
    transformation selects the largest new one-hot encoded variables and assigns it value 1 while setting the values
    of the rest of the one-hot variables to 0.

    :param x (torch tensor, <num_rows> x <num_covariates> (number of GP covariates))
    :param: GP_kernel_mapping_covar_identification (list of dicts): contains information about all covariates
        requiring special attention (name, type and which columns in train_X, train_Y)
    :return x_output (torch tensor, <num_rows> x <num_covariates> (number of GP covariates)) with transformation
        applied
    """

    x_output = x

    for mapped_covar in GP_kernel_mapping_covar_identification:

        # case where variable is of type int (integer)
        if mapped_covar["type"] == int:
            x_output[:, mapped_covar["column"]] = torch.round(
                x[:, mapped_covar["column"]]
            )

        # case where variable is of type str (categorical)
        elif mapped_covar["type"] == str:

            # identify column of max value
            _, max_index = torch.topk(x[:, mapped_covar["column"]], 1)

            # set all but column of max value to 0, max value column to 1
            # first set all entries in one-hot variables to 0
            # then run through max one-hot variable in each row to set to 1 (also handles cases of >1 observations)
            x_output[:, mapped_covar["column"]] = 0.0
            for row_id in range(max_index.size()[0]):
                x_output[
                    row_id, mapped_covar["column"][max_index[row_id, 0].item()]
                ] = 1.0

    return x_output
