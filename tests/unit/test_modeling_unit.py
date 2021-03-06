import torch
import pytest
from creative_project._modeling import _set_GP_model


@pytest.mark.parametrize("nu", [1.5, None, 2])
def test_modeling__set_GP_model_fail_unit(training_data_covar_complex, tmp_modeling_class, nu):
    """
    test that setting model fails if unsupported model_type name provided. test across a few different values for nu
    """

    covars = training_data_covar_complex[0]
    train_X = training_data_covar_complex[1]
    train_Y = training_data_covar_complex[2]

    cls = tmp_modeling_class

    # set a few key attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y

    cls.model["model_type"] = "not_supported"
    cls.model["response_sampled_iter"] = cls.train_X.shape[0]

    # run the method
    with pytest.raises(Exception) as e:
        output_text = cls._set_GP_model(nu=nu)
    assert str(e.value) == "creative_project._modeling._set_GP_model: unknown 'model_type' " \
                           "(" + cls.model["model_type"] + ") provided. Must be in following list " \
                                                            "['Custom', 'SingleTaskGP']"


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
    cls.GP_kernel_mapping_covar_identification = [
        {"type": int, "columns": [0]},
        {"type": float, "columns": [1]},
        {"type": str, "columns": [2, 3, 4]},
    ]

    # run the method
    x_output = cls._GP_kernel_transform(x_data)

    # assert result
    # gets a tensor of same size as x_output and x_result containing at each element the bool for whether the two are
    # identical
    bool_equal = (x_output == x_result)
    for j in range(bool_equal.size()[0]):
        for i in range(bool_equal.size()[1]):
            assert bool_equal[j, i].item()
