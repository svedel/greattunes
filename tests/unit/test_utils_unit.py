import pytest
import torch
from creative_project.utils import __get_covars_from_kwargs


@pytest.mark.parametrize(
    "covars",
    [
        [1.0, 2.0, 3.0],
        [1.0],
        torch.tensor([[1, 2, 3]], dtype=torch.double),
        torch.tensor([[1.56]], dtype=torch.double)
    ]
)
def test__get_covars_from_kwargs_works(covars):
    """
    positive tests of covars
    """

    covars_candidate_float_tensor = __get_covars_from_kwargs(covars)

    # cases where covars is a list
    if isinstance(covars, list):
        for i in range(len(covars)):
            assert covars_candidate_float_tensor[0, i].item() == covars[i]

    # cases where covars is a tensor
    elif isinstance(covars, torch.DoubleTensor):
        for i in range(covars.size()[1]):
            assert covars_candidate_float_tensor[0, i].item() == covars[0, i].item()


@pytest.mark.parametrize(
    "covars, error_msg",
    [
        [[1.0, 2.0, 'a'], "must be real number, not str"],
        [['b'], "too many dimensions 'str'" ],
        [torch.tensor([1.56, 12.8], dtype=torch.double), "creative_project.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'. Was expecting torch tensor of size (1,<num_covariates>) but received one of size [2]"],
    ]
)
def test__get_covars_from_kwargs_fails(covars, error_msg):
    """
    negative tests of covars
    """

    with pytest.raises(Exception) as e:
        covars_candidate_float_tensor = __get_covars_from_kwargs(covars)
    assert str(e.value) == error_msg