import pytest
import torch


@pytest.mark.parametrize("input_data",
                         [
                             [3.1, -12.2],
                             [4.5],
                         ]
                         )
def test_observe_get_and_verify_response_input_manual_functional(tmp_observe_class, input_data, monkeypatch):
    """
    test _get_and_verify_response_input for manual method for getting response values. Monkeypatching the built-in
    "input" function
    """

    # initialized temp class
    cls = tmp_observe_class
    cls.sampling["method"] = "manual"

    # set attribute
    cls.model = {"covars_proposed_iter": 0}

    # monkeypatching "input"
    monkeypatch_output = ", ".join([str(x) for x in input_data])  # match data from "input" function
    monkeypatch.setattr("builtins.input", lambda _: monkeypatch_output)

    # run function
    output = cls._get_and_verify_response_input()

    # assert
    for it in range(len(input_data)):
        assert output[0, it].item() == input_data[it]


def test_observe_get_and_verify_response_input_functions_functional(tmp_observe_class, training_data_covar_complex):
    """
    test _get_and_verify_response_input for manual method for getting response values
    """

    # initialized temp class
    cls = tmp_observe_class
    cls.sampling["method"] = "functions"
    cls.train_X = training_data_covar_complex[1]

    # add simple response function
    tmp_val = 2.2

    def mock_response_function(covar):
        """
        test response function
        :param covar: torch.tensor (num_obs X num_covariates)
        :return:
        """
        return tmp_val

    cls.sampling["response_func"] = mock_response_function

    # run test
    output = cls._get_and_verify_response_input()

    assert output[0].item() == tmp_val


def test_get_and_verify_covars_input_with_dependencies_works(tmp_observe_class, monkeypatch):
    """
    test _get_and_verify_covars_input which depends on _read_covars_manual_input and
    _validators.Validators.__validate_num_covars. monkeypatch the user input
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    covariates = [1.1, 2.2, 200, -1.7]
    covars_tensor = torch.tensor([covariates], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # define required attributes
    cls.proposed_X = covars_tensor
    cls.initial_guess = covars_tensor

    # monkeypatch
    def mock_input(x):  # mock function to replace 'input' for unit testing purposes
        return ", ".join([str(x) for x in covariates])
    monkeypatch.setattr("builtins.input", mock_input)

    # run method
    covars_candidate_float_tensor = cls._get_and_verify_covars_input()

    # assert
    for i in range(covars_candidate_float_tensor.size()[1]):
        assert covars_candidate_float_tensor[0, i].item() == covariates[i]


def test_get_and_verify_covars_input_with_dependencies_fails(tmp_observe_class, monkeypatch):
    """
    test _get_and_verify_covars_input which depends on _read_covars_manual_input and
    _validators.Validators.__validate_num_covars. monkeypatch the user input
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    covariates = [1.1, 2.2, 200, -1.7]
    covars_tensor = torch.tensor([covariates], dtype=torch.double, device=device)

    # expected data (previously proposed and initial guess)
    exp_input = covariates[:3]  # one less than the covariates to be provided above
    exp_input_tensor = torch.tensor([exp_input], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # define required attributes
    cls.proposed_X = exp_input_tensor
    cls.initial_guess = exp_input_tensor

    # monkeypatch
    def mock_input(x):  # mock function to replace 'input' for unit testing purposes
        return ", ".join([str(x) for x in covariates])
    monkeypatch.setattr("builtins.input", mock_input)

    # expected error message
    error_msg = "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like " + str(cls.proposed_X[-1]) + ", but got " + str(covars_tensor)

    # run the method, expect it to fail
    with pytest.raises(Exception) as e:
        covars_candidate_float_tensor = cls._get_and_verify_covars_input()
    assert str(e.value) == error_msg


