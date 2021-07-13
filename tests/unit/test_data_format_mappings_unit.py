import numpy as np
import pandas as pd
import pytest
import torch
from greattunes.data_format_mappings import pretty2tensor_covariate, tensor2pretty_covariate, \
    pretty2tensor_response, tensor2pretty_response


@pytest.mark.parametrize(
    "x_pandas, x_torch_return",
    [
        [pd.DataFrame({'a': [2], 'b': [3.2], 'c': ['blue']}), torch.tensor([[2.0, 3.2, 0.0, 0.0, 1.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))],
        [pd.DataFrame({'a': [2, -1], 'b': [3.2, 1.7], 'c': ['blue', 'red']}), torch.tensor([[2.0, 3.2, 0.0, 0.0, 1.0], [-1.0, 1.7, 1.0, 0.0, 0.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]
    ]
)
def test_pretty2tensor_covariate_works(covar_details_covar_mapped_names, x_pandas, x_torch_return):
    """
    test that pretty2tensor_covariate works for single and multiple observations
    """

    # get covariate details
    covar_details = covar_details_covar_mapped_names[0]
    covar_mapped_names = covar_details_covar_mapped_names[1]

    # the device for the method (for torch-stuff)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run the method
    x_torch, x_numpy = pretty2tensor_covariate(x_pandas, covar_details, covar_mapped_names, device=device)

    # check result by checking x_torch_return
    x_torch_check = (x_torch == x_torch_return)
    for j in range(x_torch_check.size()[0]):
        for i in range(x_torch_check.size()[1]):
            assert x_torch_check[j, i].item()

    # check numpy results
    x_numpy_check = (x_numpy == x_torch_return.numpy())
    for j in range(x_numpy_check.shape[0]):
        for i in range(x_numpy_check.shape[1]):
            assert x_numpy_check[j, i]


@pytest.mark.parametrize(
    "x_pandas, x_torch_return",
    [
        [pd.DataFrame({'a': [2], 'b': [3.2], 'd': [12]}), torch.tensor([[2.0, 3.2, 0.0, 0.0, 0.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))],
        [pd.DataFrame({'a': [2, -1], 'b': [3.2, 1.7], 'd': [12, 15]}), torch.tensor([[2.0, 3.2, 0.0, 0.0, 0.0], [-1.0, 1.7, 0.0, 0.0, 0.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]
    ]
)
def test_pretty2tensor_covariate_adds_missing_columns_removes_extra_columns(covar_details_covar_mapped_names, x_pandas, x_torch_return):
    """
    checks that columns are added to outcome (x_output) for variables even if said variable is not present in input
    x_data (pandas). In these examples are expecting a categorical variable with three elements (red, green, blue) to
    be present in data as well.

    also checks that additional columns not needed (as specified by covar_details) are removed
    """

    # get covariate details
    covar_details = covar_details_covar_mapped_names[0]
    covar_mapped_names = covar_details_covar_mapped_names[1]

    # the device for the method (for torch-stuff)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run the method
    x_torch, x_numpy = pretty2tensor_covariate(x_pandas, covar_details, covar_mapped_names, device=device)

    # check result by checking x_torch_return
    x_torch_check = (x_torch == x_torch_return)
    for j in range(x_torch_check.size()[0]):
        for i in range(x_torch_check.size()[1]):
            assert x_torch_check[j, i].item()

    # check numpy results
    x_numpy_check = (x_numpy == x_torch_return.numpy())
    for j in range(x_numpy_check.shape[0]):
        for i in range(x_numpy_check.shape[1]):
            assert x_numpy_check[j, i]


@pytest.mark.parametrize(
    "train_X_sample, pandas_out",
    [
        [torch.tensor([[2.0, 3.2, 0.0, 0.0, 1.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({'a': [2], 'b': [3.2], 'c': ['blue']})],
        [torch.tensor([[2.0, 3.2, 0.0, 0.0, 1.0], [-1.0, 1.7, 1.0, 0.0, 0.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({'a': [2, -1], 'b': [3.2, 1.7], 'c': ['blue', 'red']})],
    ]
)
def test_tensor2pretty_covariate_works(covar_details_covar_mapped_names, train_X_sample, pandas_out):
    """
    tests that the reverse mapping tensor2pretty_covariate works for single and multiple observations
    """

    # get covariate details
    covar_details = covar_details_covar_mapped_names[0]

    # run the method
    x_pandas = tensor2pretty_covariate(train_X_sample, covar_details)

    # compare
    pd_bool = (x_pandas == pandas_out).values
    for j in range(pd_bool.shape[0]):
        for i in range(pd_bool.shape[1]):
            assert pd_bool[j, i]


@pytest.mark.parametrize(
    "y_pandas",
    [
        pd.DataFrame({"Response": [1.0, 2.0, 3.0]}), pd.DataFrame({"Response": [22.3]})
    ]
)
def test_pretty2tensor_response(y_pandas):
    """
    test that the mapping to tensor format works
    """

    # run the function
    tensor_out = pretty2tensor_response(y_pandas=y_pandas)

    # compare
    for i in range(y_pandas.shape[0]):
        assert tensor_out[i,0].item() == y_pandas["Response"].iloc[i]


@pytest.mark.parametrize(
    "y_tensor",
    [
        torch.tensor([[1, 2, 3]],dtype=torch.double), torch.tensor([[22.3]],dtype=torch.double)
    ]
)
def test_tensor2pretty_response(y_tensor):
    """
    test that the mapping back from tensor to pandas works
    """

    # run mapping
    pandas_out = tensor2pretty_response(train_Y_sample=y_tensor)

    # compare
    for i in range(y_tensor.size()[0]):
        assert y_tensor[i,0].item() == pandas_out["Response"].iloc[i]

