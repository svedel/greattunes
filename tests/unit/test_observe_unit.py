import pytest
import torch

@pytest.mark.parametrize("method, tmp_val",
                         [
                             ["functions", 1.0],
                             ["manual", 2.0]
                         ])
def test_observe_get_and_verify_response_input_unit(tmp_observe_class, method, tmp_val, monkeypatch):
    """
    test that _get_and_verify_response_input works for self.sampling["method"] = "manual" or "functions". Leverage
    monkeypatching and create false class to mock that creative_project._observe will be called inside
    CreativeProject class in creative_project.__init__
    """

    # # define class
    cls = tmp_observe_class
    cls.sampling["method"] = method

    # monkeypatch the "support" functions _get_response_function_input, _read_response_manual_input
    def mock_get_response_function_input():
        return torch.tensor([[tmp_val]], dtype=torch.double,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    monkeypatch.setattr(
        cls, "_get_response_function_input", mock_get_response_function_input
    )

    manual_tmp_val = tmp_val + 1.0
    def mock_read_response_manual_input(additional_text):
        return torch.tensor([[manual_tmp_val]], dtype=torch.double,
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    monkeypatch.setattr(
        cls, "_read_response_manual_input", mock_read_response_manual_input
    )

    # run test
    output = cls._get_and_verify_response_input()

    if method == "functions":
        assert output[0].item() == tmp_val
    elif method == "manual":
        assert output[0].item() == manual_tmp_val


@pytest.mark.parametrize("method", ["WRONG", None])
def test_observe_get_and_verify_response_input_fail_unit(tmp_observe_class, method):
    """
    test that _get_and_verify_response_input fails for self.sampling["method"] not equal to "manual" or "functions".
    """

    # # define class
    cls = tmp_observe_class
    cls.sampling["method"] = method

    with pytest.raises(Exception) as e:
        assert output == cls._get_and_verify_response_input()
    assert str(e.value) == "creative_project._observe._get_and_verify_response_input: class attribute " \
                           "self.sampling['method'] has non-permissable value " + str(method) + ", must be in " \
                           "['manual', 'functions']."


@pytest.mark.parametrize("FLAG_TRAINING_DATA", [True, False])
def test_observe_get_response_function_input_unit(tmp_observe_class, training_data_covar_complex, FLAG_TRAINING_DATA):
    """
    test _get_response_function_input for pass and fail
    """

    # temp class for test
    cls = tmp_observe_class

    # data
    train_X = training_data_covar_complex[1]

    # set attributes on class, required for test
    cls.train_X = None
    if FLAG_TRAINING_DATA:
        cls.train_X = train_X

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

    # assert
    if FLAG_TRAINING_DATA:
        # run test
        output = cls._get_response_function_input()

        assert output[0].item() == tmp_val
    else:
        with pytest.raises(Exception) as e:
            assert output == cls._get_response_function_input()
        assert str(e.value) == "'NoneType' object is not subscriptable"


@pytest.mark.parametrize("additional_text, input_data",
                         [
                             ["temp", [1.1, 2.2]],
                             ["try again", [3.1, -12.2]],
                             ["simple try", [4.5]],
                         ]
                         )
def test_observe_read_response_manual_input_unit(tmp_observe_class, additional_text, input_data, monkeypatch):
    """
    test _read_response_manual_input, monkeypatching the "input" function call in the method
    """

    # temp class for test
    cls = tmp_observe_class

    # set attribute
    cls.model = {"covars_proposed_iter": 0}

    # monkeypatching "input"
    monkeypatch_output = ", ".join([str(x) for x in input_data])  # match data from "input" function
    monkeypatch.setattr("builtins.input", lambda _: monkeypatch_output)

    # run function
    output = cls._read_response_manual_input(additional_text)

    # assert
    for it in range(len(input_data)):
        assert output[0, it].item() == input_data[it]

@pytest.mark.parametrize(
    "candidate",
    [
        torch.tensor([[2.2]], dtype=torch.double),
        torch.tensor([[2.2, 3.3, -1]], dtype=torch.double),
    ]
)
def test_observe_print_candidate_to_prompt_works_unit(tmp_observe_class, candidate):
    """
    test that given a candidate, the right string is written by the method _print_candidate_to_prompt
    :param candidate (torch tensor): one-row tensor of new datapoint to be investigated
    """

    # temporary class to run the test
    cls = tmp_observe_class

    # extend with required attributes
    tmp_covars_proposed_iter = 2
    cls.model = {"covars_proposed_iter": tmp_covars_proposed_iter}

    # run the method: generate the string to be printed
    input_request = cls._print_candidate_to_prompt(candidate=candidate)

    # build expected output
    first = True
    outtext = "ITERATION " + str(tmp_covars_proposed_iter) + " - NEW datapoint to sample: "
    for tmp in candidate[0]:
        if first:
            outtext += str(tmp.item())
            first = False
        else:
            outtext += ", " + str(tmp.item())

    # assert
    assert input_request == outtext


@pytest.mark.parametrize(
    "candidate, error_msg",
    [
        [torch.tensor([], dtype=torch.double), "kre8_core.creative_project._observe._print_candidate_to_prompt: provided input 'candidate' is empty. Expecting torch tensor of size 1 X num_covariates"],
        [None, "kre8_core.creative_project._observe._print_candidate_to_prompt: provided input 'candidate' is incorrect datatype. Expecting to be of type torch.Tensor"]
    ]
)
def test_observe_print_candidate_to_prompt_fails_unit(tmp_observe_class, candidate, error_msg):
    """
    test that _print_candidate_to_prompt throws the right error for the two cases
    :param candidate: supposed to be one-row tensor of new datapoint to be investigated of type torch tensor, here hijacking
    """

    # temporary class to run the test
    cls = tmp_observe_class

    # run _print_candidate_to_prompt method and ensure correct error returned
    with pytest.raises(Exception) as e:
        # run the method: generate the string to be printed
        input_request = cls._print_candidate_to_prompt(candidate=candidate)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "additional_text", ["testing function", "12345_ygh", None, 22.0, [1.0, 4.4], torch.tensor([[2.2]], dtype=torch.double)]
)
def test_read_covars_manual_input(tmp_observe_class, additional_text, monkeypatch):
    """
    test reading of covars from manual input by user. Monkeypatches reliance on function 'input'
    """

    covariates = [1.1, 2.2, 200, -1.7]

    # temp class to execute the test
    cls = tmp_observe_class

    # add attribute 'initial_guess' required for '_read_covars_manual'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls.initial_guess = torch.tensor([covariates], dtype=torch.double, device=device)

    # monkeypatch
    def mock_input(x):  # mock function to replace 'input' for unit testing purposes
        return ", ".join([str(x) for x in covariates])
    monkeypatch.setattr("builtins.input", mock_input)

    # run the test
    # different tests for cases where it's supposed to pass vs fail
    if isinstance(additional_text, str):
        covars_candidate_float_tensor = cls._read_covars_manual_input(additional_text)

        # assert that the right elements are returned in 'covars_candidate_float_tensor'
        for i in range(covars_candidate_float_tensor.size()[1]):
            assert covars_candidate_float_tensor[0,i].item() == covariates[i]

    # cases where type of additonal_text should make test fail
    else:
        with pytest.raises(AssertionError) as e:
            covars_candidate_float_tensor = cls._read_covars_manual_input(additional_text)
        assert str(e.value) == "creative_project._observe._read_covars_manual_input: wrong datatype of parameter 'additional_text'. Was expecting 'str' but received " + str(type(additional_text))