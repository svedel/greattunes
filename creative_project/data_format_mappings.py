import numpy as np
import pandas as pd
import torch

"""
mapping between the pretty format which users understand (named variables) and the format used internally in the
framework by the Gaussian Process models (torch tensor, double data type and one-hot encoding for categorical
variables)
"""

def pretty2tensor(x_pandas, covar_details, covar_mapped_names, device=None):
    """
    maps between the pretty format (based on pandas) and to the tensor format used behind the scenes

    :param x_pandas (data frame): dataframe containing one or multiple covariate observations (one or multiple rows) in
    the natural format (i.e. for categorical variables give just the categorical observation without the full range of
    possible outcomes)
    :param covar_details (dict of dicts): ...
    :param covar_mapped_names (list of str): contains the names of all columns and order of all covariates in the tensor
    format used behind the scenes (incl one-hot encoded columns)
    :param device (torch.device; default None): the device for torch calculations determining whether CPU or GPU is
    used. Uses device of local machine if 'None' provided
    :return mapped covariates in tensor format (torch)
    :return  mapped covariates in array format (numpy)
    """

    # list of all covariates reported in covar_details
    names = covar_mapped_names

    # === adds covariates missing from x_pandas but provided previously ===

    # looks whether data for all required covariates is present
    x_pandas_tmp = x_pandas
    covar_names = list(x_pandas.columns)
    miss_cols = set(names).difference(covar_names)

    # add missing columns from covar_details to x_pandas as 0
    for newcol in miss_cols:
        x_pandas_tmp[newcol] = 0.0

    # === map categorical variables to one-hot encoding, add as columns ===

    # identify the categorical covariates
    for covar_key in covar_details:
        if covar_details[covar_key]["type"] == str:
            # checks whether this covariate is present as categorical variable in the user-provided 'x_pandas'
            # if this is not the case, will be setting it to zero in all cases
            if covar_key in covar_names:
                # get the value for each column in x_pandas
                tmp_cat_data = x_pandas[covar_key].values

                # set the value to 1 for the appropriate one-hot encoded value
                for i in range(len(tmp_cat_data)):
                    cat_val = tmp_cat_data[i]

                    # one-hot column
                    one_hot_column = covar_key + "_" + cat_val

                    # update the right one-hot encoded column
                    x_pandas_tmp.loc[x_pandas_tmp[covar_key] == cat_val, one_hot_column] = 1.0

    # sort according to order in covar_details and pick only the relevant columns
    x_pandas_tmp = x_pandas_tmp[covar_mapped_names]

    # sets 'device' for torch tensor
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # return in torch tensor and numpy array formats
    return torch.tensor(x_pandas_tmp.values, dtype=torch.double, device=device), x_pandas_tmp.to_numpy()