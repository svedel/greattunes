import pytest
import torch
from creative_project.transformed_kernel_models.transformation import GP_kernel_transform

@pytest.mark.parametrize(
    "x_data, x_result",
    [
        [torch.tensor([[1.2, 2.2, 0.7, 0.3, 0.6]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[1.0, 2.2, 1.0, 0.0, 0.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))],
        [torch.tensor([[1.0, 2.2, 0.7, 0.3, 0.6], [2.7, -1.2, 0.2, 0.6, 0.5]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[1.0, 2.2, 1.0, 0.0, 0.0], [3.0, -1.2, 0.0, 1.0, 0.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))]
     ]
)
def test_modeling_GP_kernel_transform(tmp_modeling_class, x_data, x_result):
    """
    test that _GP_kernel_transform works for float, int and categorical variables (represented by str).
    provides covariate tensor data samples as input, as well as the expected result after running the method.
    """

    cls = tmp_modeling_class

    # set some attributes
    # define a covariate vector where the first element is of type int, second is a float, and the third is categorical
    # with the options ("red", "green", "blue")
    GP_kernel_mapping_covar_identification = [
        {"type": int, "column": [0]},
        {"type": float, "column": [1]},
        {"type": str, "column": [2, 3, 4]},
    ]

    # run the method
    x_output = GP_kernel_transform(x_data, GP_kernel_mapping_covar_identification)

    # assert result
    # gets a tensor of same size as x_output and x_result containing at each element the bool for whether the two are
    # identical
    bool_equal = (x_output == x_result)
    for j in range(bool_equal.size()[0]):
        for i in range(bool_equal.size()[1]):
            assert bool_equal[j, i].item()