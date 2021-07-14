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

    x_output = x.clone()

    for mapped_covar in GP_kernel_mapping_covar_identification:

        # case where variable is of type int (integer)
        if mapped_covar["type"] == int:
            x_output[..., mapped_covar["column"]] = torch.round(
                x[..., mapped_covar["column"]]
            )

        # case where variable is of type str (categorical)
        elif mapped_covar["type"] == str:

            # identify column of max value
            _, max_index = torch.topk(
                x[..., mapped_covar["column"]], 1, dim=-1
            )  # apply row-wise (at dim -2)

            # set all but column of max value to 0, max value column to 1
            # first determine indices for which entries to mark as largest entries in colums identified by
            # mapped_covar["column"]
            if len(max_index.size()) == 2:  # input data x is a rank 2 tensor
                # row indices
                d1_indices = torch.tensor([range(max_index.size()[-2])])
                # column indices
                d2_indices = max_index.flatten()
                indices = (d1_indices, d2_indices)
            elif len(max_index.size()) == 3:  # input data x is a rank 3 tensor
                # indices in 3rd dimension
                d1_indices = torch.tensor(
                    [[i] * max_index.size()[-2] for i in range(max_index.size()[-3])]
                ).flatten()
                # row indices
                d2_indices = torch.tensor(
                    [range(max_index.size()[-2])] * max_index.size()[-3]
                ).flatten()
                # column indices
                d3_indices = max_index.flatten()
                indices = (d1_indices, d2_indices, d3_indices)
            else:
                raise Exception(
                    "create_project.transformed_kernel_models.transformation.GP_kernel_transform: provided "
                    "input data 'x' is a tensor of rank 4; currently on tensors of ranks 2 and 3 are"
                    "supported."
                )

            # set max val entries to 1, ignore duplicates
            tmp = torch.zeros(
                [dsize for dsize in x.size()[:-1]] + [len(mapped_covar["column"])],
                dtype=torch.double,
            )
            x_output[..., mapped_covar["column"]] = tmp.index_put(
                indices=indices, values=torch.tensor([1.0], dtype=torch.double)
            )

    return x_output
