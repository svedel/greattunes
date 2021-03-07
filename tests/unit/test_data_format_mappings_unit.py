import numpy as np
import pandas as pd
import pytest
import torch
from creative_project.data_format_mappings import pretty2tensor, tensor2pretty


@pytest.mark.parametrize(
    "x_pandas, x_torch_return",
    [
        [pd.DataFrame({'a': [2], 'b': [3.2], 'c': ['blue']}), torch.tensor([[2.0, 3.2, 0.0, 0.0, 1.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))],
        [pd.DataFrame({'a': [2, -1], 'b': [3.2, 1.7], 'c': ['blue', 'red']}), torch.tensor([[2.0, 3.2, 0.0, 0.0, 1.0], [-1.0, 1.7, 1.0, 0.0, 0.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]
    ]
)
def test_pretty2tensor_works(covar_details_covar_mapped_names, x_pandas, x_torch_return):
    """
    test that pretty2tensor works for single and multiple observations
    """

    # get covariate details
    covar_details = covar_details_covar_mapped_names[0]
    covar_mapped_names = covar_details_covar_mapped_names[1]

    # the device for the method (for torch-stuff)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run the method
    x_torch, x_numpy = pretty2tensor(x_pandas, covar_details, covar_mapped_names, device=device)

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
def test_pretty2tensor_adds_missing_columns_removes_extra_columns(covar_details_covar_mapped_names, x_pandas, x_torch_return):
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
    x_torch, x_numpy = pretty2tensor(x_pandas, covar_details, covar_mapped_names, device=device)

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
def test_tensor2pretty_works(covar_details_covar_mapped_names, train_X_sample, pandas_out):
    """
    tests that the reverse mapping tensor2pretty works for single and multiple observations
    """

    # get covariate details
    covar_details = covar_details_covar_mapped_names[0]

    # run the method
    x_pandas = tensor2pretty(train_X_sample, covar_details)

    # compare
    pd_bool = (x_pandas == pandas_out).values
    for j in range(pd_bool.shape[0]):
        for i in range(pd_bool.shape[1]):
            assert pd_bool[j, i]
