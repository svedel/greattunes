import pytest
import pandas as pd
import torch
from creative_project.data_format_mappings import pretty2tensor_covariate, tensor2pretty_covariate

@pytest.mark.parametrize(
    "x_pandas",
    [pd.DataFrame({'a': [2], 'b': [3.2], 'c': ['blue']}), pd.DataFrame({'a': [2, -1], 'b': [3.2, 1.7], 'c': ['blue', 'red']})]
)
def test_pretty2tensor_then_tensor2pretty(covar_details_covar_mapped_names, x_pandas):
    """
    test that mapping back and forth between pandas and tensor formats is reversible
    """

    # get covariate details
    covar_details = covar_details_covar_mapped_names[0]
    covar_mapped_names = covar_details_covar_mapped_names[1]

    # the device for the method (for torch-stuff)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run the forward mapping
    x_torch, x_numpy = pretty2tensor_covariate(x_pandas, covar_details, covar_mapped_names, device=device)

    # run the reverse mapping
    x_pandas_returned = tensor2pretty_covariate(x_torch, covar_details)

    # compare
    pd_bool = (x_pandas == x_pandas_returned).values
    for j in range(pd_bool.shape[0]):
        for i in range(pd_bool.shape[1]):
            assert pd_bool[j, i]
