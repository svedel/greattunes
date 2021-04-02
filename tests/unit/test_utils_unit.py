import pytest
import torch
from creative_project.utils import __get_covars_from_kwargs, DataSamplers


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
        [torch.tensor([1.56, 12.8], dtype=torch.double), "creative_project.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'. Was expecting torch tensor of size (1,<num_covariates>) but received one of size (2)."],
    ]
)
def test__get_covars_from_kwargs_fails(covars, error_msg):
    """
    negative tests of covars
    """

    with pytest.raises(Exception) as e:
        covars_candidate_float_tensor = __get_covars_from_kwargs(covars)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "initial_guess, covar_bounds, n_samp",
    [
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.5, 1.8, 2],[1.5, 2.8, 3.9]], dtype=torch.double), 1],
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.5, 1.8, 2],[1.5, 2.8, 3.9]], dtype=torch.double), 3],
    ]
)
def test_DataSamplers_random(initial_guess, covar_bounds, n_samp):
    """
    test that DataSamplers.random generates a set of random datapoints within the limits provided by covar_bounds and
    with the right number of rows (should be n_samp) and columns (should be same as number of entries in initial_guess
    for the 'random' method
    :param initial_guess:
    :param covar_bounds:
    :param n_samp:
    :return:
    """

    # number of covariates
    NUM_COVAR = initial_guess.shape[1]

    # define the class
    cls = DataSamplers()

    # set additional attribute device used to cast to right computational device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate the candidates
    candidates = DataSamplers().random(n_samp=n_samp, initial_guess=initial_guess, covar_bounds=covar_bounds,
                                       device=device)

    # assert the size
    assert candidates.size()[0] == n_samp
    assert candidates.size()[1] == NUM_COVAR

    # assert the content is within bounds
    for i in range(NUM_COVAR):
        for j in range(n_samp):
            assert (candidates[j,i].item() >= covar_bounds[0,i].item())&(candidates[j,i].item() <= covar_bounds[1,i].item())


@pytest.mark.parametrize(
    "initial_guess, covar_bounds, n_samp",
    [
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.5, 1.8, 2],[1.5, 2.8, 3.9]], dtype=torch.double), 1],
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.5, 1.8, 2],[1.5, 2.8, 3.9]], dtype=torch.double), 3],
    ]
)
def test_DataSamplers_latin_hcs(initial_guess, covar_bounds, n_samp):
    """
    test that DataSamplers.random generates a set of random datapoints within the limits provided by covar_bounds and
    with the right number of rows (should be n_samp) and columns (should be same as number of entries in initial_guess
    for the 'latin_hcs' method
    :param initial_guess:
    :param covar_bounds:
    :param n_samp:
    :return:
    """

    # number of covariates
    NUM_COVAR = initial_guess.shape[1]

    # define the class
    cls = DataSamplers()

    # set additional attribute device used to cast to right computational device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate the candidates
    candidates = DataSamplers().latin_hcs(n_samp=n_samp, initial_guess=initial_guess, covar_bounds=covar_bounds,
                                       device=device)

    # assert the size
    assert candidates.size()[0] == n_samp
    assert candidates.size()[1] == NUM_COVAR

    # assert the content is within bounds
    for i in range(NUM_COVAR):
        for j in range(n_samp):
            assert (candidates[j,i].item() >= covar_bounds[0,i].item())&(candidates[j,i].item() <= covar_bounds[1,i].item())
