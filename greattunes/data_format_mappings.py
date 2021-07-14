"""
mapping between the pretty format which users understand (named variables) and the format used internally in the
framework by the Gaussian Process models (torch tensor, double data type and one-hot encoding for categorical
variables)
"""
import pandas as pd
import torch


def pretty2tensor_covariate(x_pandas, covar_details, covar_mapped_names, device=None):
    """
    maps between the pretty format (based on pandas) and to the tensor format used behind the scenes. This is
    the reverse mapping to 'tensor2pretty_covariate'

    :param x_pandas (data frame): dataframe containing one or multiple covariate observations (one or multiple rows) in
    the natural format (i.e. for categorical variables give just the categorical observation without the full range of
    possible outcomes)
    :param covar_details (dict of dicts): ...
    :param covar_mapped_names (list of str): contains the names of all columns and order of all covariates in the
        tensor format used behind the scenes (incl one-hot encoded columns)
    :param device (torch.device; default None): the device for torch calculations determining whether CPU or GPU is
    used. Uses device of local machine if 'None' provided
    :return mapped covariates in tensor format (torch)
    :return mapped covariates in array format (numpy)
    """

    # list of all covariates reported in covar_details
    names = covar_mapped_names

    # === adds covariates missing from x_pandas but provided previously ===

    # looks whether data for all required covariates is present
    x_pandas_tmp = x_pandas.copy()
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
                    x_pandas_tmp.loc[
                        x_pandas_tmp[covar_key] == cat_val, one_hot_column
                    ] = 1.0

    # sort according to order in covar_details and pick only the relevant columns
    x_pandas_tmp = x_pandas_tmp[covar_mapped_names]

    # sets 'device' for torch tensor
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # return in torch tensor and numpy array formats
    return (
        torch.tensor(x_pandas_tmp.values, dtype=torch.double, device=device),
        x_pandas_tmp.to_numpy(),
    )


# reverse mapping (from internal format to user-readable format)
def tensor2pretty_covariate(train_X_sample, covar_details):
    """
    maps between the tensor format used behind the scenes and the user-facing pretty format (based on pandas). This is
    the reverse mapping to 'pretty2tensor_covariate'

    :param train_X_sample (torch tensor in format <num_observations> x <num_covars>):
    :param covar_details (dict of dicts): ...
    :return:
    """
    # check that covar_details has been populated
    if len(covar_details) == 0:
        raise Exception(
            "greattunes.data_format_mappings.tensor2pretty_covariate: class instance has not been properly "
            "initialized, 'covar_details' has not been initiated"
        )

    # map back
    keys = list(covar_details.keys())
    dict_out = {}

    for i in range(len(covar_details)):

        # for case where covariate is of type int
        if covar_details[keys[i]]["type"] == int:

            # get right column in train_X
            # use covar_details[key]["columns"] to extract column info. This is an integer for data type 'int'
            colnum = covar_details[keys[i]]["columns"]

            # get the data element to show the user
            # convert from 'double' format in 'train_X' to integer
            elements = [
                int(tensor_iter.item())
                for tensor_iter in train_X_sample[:, colnum].round()
            ]

            # update output
            dict_out[keys[i]] = elements

        # for case where covariate is of type str (categorical)
        elif covar_details[keys[i]]["type"] == str:

            # get right columns in train_X
            # use covar_details[key]["columns"] to extract column info. This is an list of integers for data type
            # 'str' (categorical variable) since we are using one-hot encoding. Only pick category with largest value
            colnums = covar_details[keys[i]]["columns"]

            # get max value from one-hot encoding columns for each row
            # max_index is a tensor of size num_observations x 1 containing the entry in train_X_sample[:, colnums]
            # that has the highest value at each row
            _, max_index = torch.topk(train_X_sample[:, colnums], 1)

            # get the categorical names for the max one-hot column at each row
            # the maximum value across the columns in 'colnums' for each row is in max_index (torch tensor of type int)
            # covar_details[<key>]["options"] contains the options of the categorical variable, while the entry
            # ["opt_names"] contains the names of the columns for the one-hot encoded variables in format
            # <variable_name>_<option_name>. We get the correct values by removing the prefix <variable_name>_ from
            # opt_names
            prefix_name = keys[i] + "_"
            prefix_length = len(prefix_name)
            cat_names = [
                name[prefix_length:] for name in covar_details[keys[i]]["opt_names"]
            ]

            elements = [cat_names[cat_iter.item()] for cat_iter in max_index]

            # update output
            dict_out[keys[i]] = elements

        # for case where covariate is of type float
        elif covar_details[keys[i]]["type"] == float:

            # get right column in train_X
            # use covar_details[key]["columns"] to extract column info. This is an integer for data type 'int'
            colnum = covar_details[keys[i]]["columns"]

            # get the data element to show the user
            elements = [tensor_iter.item() for tensor_iter in train_X_sample[:, colnum]]

            # update output
            dict_out[keys[i]] = elements

    # output as pandas dataframe
    df_out = pd.DataFrame.from_dict(dict_out)

    return df_out


def pretty2tensor_response(y_pandas, device=None):
    """
    maps between the pretty format for the response "y_pandas" and the tensor format used behind the scenes. Accepts
    only continuous response variables

    if "device"=None, then the device of the current system will be assigned
    :param y_pandas (single-row data frame): response recordings
    :return: tensor_out (torch tensor, size <num_observations> X 1)
    """

    # set the device if None provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the data
    # casting as float64 datatype translates to the same as "torch.double"
    tensor_out = torch.from_numpy(y_pandas.values.astype("float64")).to(device)

    return tensor_out


def tensor2pretty_response(train_Y_sample):
    """
    maps between the tensor format used behind the scenes and the pretty format for the response "y_pandas"

    :param train_Y_sample (tensor, size <num_observations> X 1):
    :return: y_pandas
    """

    # get the data
    resp_list = [i[0] for i in train_Y_sample.numpy()]
    y_pandas = pd.DataFrame({"Response": resp_list})

    return y_pandas
