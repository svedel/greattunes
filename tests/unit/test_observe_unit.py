import pytest
import torch
import creative_project.utils
from creative_project.data_format_mappings import tensor2pretty_covariate

@pytest.mark.parametrize("method, tmp_val",
                         [
                             ["functions", 1.0],
                             ["iterative", 2.0]
                         ])
def test_observe_get_and_verify_response_input_unit(tmp_observe_class, method, tmp_val, monkeypatch):
    """
    test that _get_and_verify_response_input works for self.sampling["method"] = "iteratuve" or "functions". Leverage
    monkeypatching and create false class to mock that creative_project._observe will be called inside
    CreativeProject class in creative_project.__init__. Rely on manual input for "iterative" option
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

    # set kwarg response to None (so manually provided input is used)
    kwarg_response = None

    # run test
    output = cls._get_and_verify_response_input(response=kwarg_response)

    if method == "functions":
        assert output[0].item() == tmp_val
    elif method == "iterative":
        assert output[0].item() == manual_tmp_val


@pytest.mark.parametrize("method", ["WRONG", None])
def test_observe_get_and_verify_response_input_fail_unit(tmp_observe_class, method):
    """
    test that _get_and_verify_response_input fails for self.sampling["method"] not equal to "iterative" or "functions".
    """

    # # define class
    cls = tmp_observe_class
    cls.sampling["method"] = method

    # set kwarg response to None (so manually provided input is used)
    kwarg_response = None

    with pytest.raises(Exception) as e:
        assert output == cls._get_and_verify_response_input(response=kwarg_response)
    assert str(e.value) == "creative_project._observe._get_and_verify_response_input: class attribute " \
                           "self.sampling['method'] has non-permissable value " + str(method) + ", must be in " \
                           "['iterative', 'functions']."


@pytest.mark.parametrize(
    "kwarg_response",
    [
        [1.2],
        torch.tensor([[1.2]], dtype=torch.double)
    ]
)
def test_get_and_verify_response_input_kwarg_input_works(tmp_observe_class, kwarg_response, monkeypatch):
    """
    test that _get_and_verify_response_input works for self.sampling["method"] = "iterative" with programmatically
    provided input. Leverage monkeypatching for utils.__get_covars_from_kwargs and create false class to mock that
    creative_project._observe will be called inside CreativeProject class in creative_project.__init__
    """

    # set device for torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # define class
    cls = tmp_observe_class
    cls.sampling["method"] = "iterative"

    # monkeypatch "__get_covars_from_kwargs"
    def mock__get_covars_from_kwargs(x):
        if isinstance(kwarg_response, list):
            return torch.tensor([kwarg_response], dtype=torch.double, device=device)
        else:
            return kwarg_response
    monkeypatch.setattr(creative_project.utils, "__get_covars_from_kwargs", mock__get_covars_from_kwargs)

    # run test
    output = cls._get_and_verify_response_input(response=kwarg_response)

    # assert
    if isinstance(kwarg_response, list):
        assert output[0].item() == kwarg_response[0]
    elif isinstance(kwarg_response, torch.DoubleTensor):
        assert output[0].item() == kwarg_response[0].item()


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


@pytest.mark.parametrize(
    "response, kwarg_response, error_msg",
    [
        [torch.tensor([[2]], dtype=torch.double), ['a'], "too many dimensions 'str'"],
        [torch.tensor([[2]], dtype=torch.double), [1, 2], "creative_project._observe._get_and_verify_response_input: incorrect number of variables provided. Was expecting input of size (1,1) but received torch.Size([1, 2])"],
        [torch.tensor([[2]], dtype=torch.double), [1, 'a'], "must be real number, not str"],
        [torch.tensor([[2, 3]], dtype=torch.double), None, "creative_project._observe._get_and_verify_response_input: incorrect number of variables provided. Was expecting input of size (1,1) but received torch.Size([1, 2])"],
        [torch.tensor([[2]], dtype=torch.double), torch.tensor([[1, 2]], dtype=torch.double), "creative_project._observe._get_and_verify_response_input: incorrect number of variables provided. Was expecting input of size (1,1) but received torch.Size([1, 2])"],
    ]
)
def test_get_and_verify_response_input_fails_wrong_input(tmp_observe_class, response, kwarg_response, error_msg,
                                                         monkeypatch):
    """
    test that _get_and_verify_response_input fails for wrong inputs. Use only the "iterative" sampling option for this
    test
    """

    # set device for torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # define class
    cls = tmp_observe_class
    cls.sampling["method"] = "iterative"

    # monkeypatch "__get_covars_from_kwargs"
    def mock__get_covars_from_kwargs(x):
        if isinstance(kwarg_response, list):
            return torch.tensor([kwarg_response], dtype=torch.double, device=device)
        else:
            return kwarg_response
    monkeypatch.setattr(creative_project.utils, "__get_covars_from_kwargs", mock__get_covars_from_kwargs)

    # monkeypatch _read_response_manual_input
    def mock_read_response_manual_input(additional_text):
        return response
    monkeypatch.setattr(
        cls, "_read_response_manual_input", mock_read_response_manual_input
    )

    # run test
    with pytest.raises(Exception) as e:
        output = cls._get_and_verify_response_input(response=kwarg_response)
    assert str(e.value) == error_msg


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

    # add covariate details to tmp_observe_class
    covar_details = {}
    for i in range(candidate.size()[1]):
        key = "covar" + str(i)
        val = candidate[0,i].item()
        covar_details[key] = {"guess": val, "min": val-1.0, "max": val+1.0, "type": float, "columns": i}
    cls.covar_details = covar_details

    # run the method: generate the string to be printed
    input_request = cls._print_candidate_to_prompt(candidate=candidate)

    # build expected output
    cand_pretty = tensor2pretty_covariate(train_X_sample=candidate, covar_details=covar_details)
    new_cand_names = [i + " (" + str(covar_details[i]["type"]) + ")" for i in list(cand_pretty.columns)]
    cand_pretty.columns = new_cand_names

    outtext = "\tNEW datapoint to sample:\n\t" + cand_pretty.to_string(index=False)

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
    covar_tensor = torch.tensor([covariates], dtype=torch.double, device=device)
    cls.initial_guess = covar_tensor

    # add proposed_X attributed required for '_read_covars_manual'
    cls.proposed_X = covar_tensor

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
            assert covars_candidate_float_tensor[0, i].item() == covariates[i]

    # cases where type of additonal_text should make test fail
    else:
        with pytest.raises(AssertionError) as e:
            covars_candidate_float_tensor = cls._read_covars_manual_input(additional_text)
        assert str(e.value) == "creative_project._observe._read_covars_manual_input: wrong datatype of parameter 'additional_text'. Was expecting 'str' but received " + str(type(additional_text))



def test_get_and_verify_covars_input_works(tmp_observe_class, monkeypatch):
    """
    test that _get_and_verify_covars_input works when providing the correct data. Monkeypatching methods
    "_read_covars_manual_input" and "__validate_num_covars"
    """

    # covariates to sample
    covariates = [1.1, 2.2, 200, -1.7]

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # temp class to execute the test
    cls = tmp_observe_class

    # monkeypatch "_read_covars_manual_input"
    def mock_read_covars_manual_input(x):
        return torch.tensor([covariates], dtype=torch.double, device=device)
    monkeypatch.setattr(cls, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_Validators__validate_num_covars"
    def mock_Validators__validate_num_covars(x):
        return True
    monkeypatch.setattr(cls, "_Validators__validate_num_covars", mock_Validators__validate_num_covars)

    # covariate kwargs is set to None so input-based method is used
    kwarg_covariates = None

    # run method
    covars_candidate_float_tensor = cls._get_and_verify_covars_input(covars=kwarg_covariates)

    # assert the output
    # assert that the right elements are returned in 'covars_candidate_float_tensor'
    for i in range(covars_candidate_float_tensor.size()[1]):
        assert covars_candidate_float_tensor[0, i].item() == covariates[i]


@pytest.mark.parametrize(
    "covars",
    [
        [1.1, 2.2, 200, -1.7],
        torch.tensor([[1.1, 2.2, 200, -1.7]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    ]

)
def test_get_and_verify_covars_programmatic_works(tmp_observe_class, covars, monkeypatch):
    """
    test that _get_and_verify_covars_input works when providing the correct data programmatically. Monkeypatching
    method "__validate_num_covars" and helper function "utils.__get_covars_from_kwargs"
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # temp class to execute the test
    cls = tmp_observe_class

    # monkeypatch "__get_covars_from_kwargs"
    def mock__get_covars_from_kwargs(x):
        if isinstance(covars, list):
            return torch.tensor([covars], dtype=torch.double, device=device)
        else:
            return covars
    monkeypatch.setattr(creative_project.utils, "__get_covars_from_kwargs", mock__get_covars_from_kwargs)

    # monkeypatch "_Validators__validate_num_covars"
    def mock_Validators__validate_num_covars(x):
        return True
    monkeypatch.setattr(cls, "_Validators__validate_num_covars", mock_Validators__validate_num_covars)

    # run method
    covars_candidate_float_tensor = cls._get_and_verify_covars_input(covars=covars)

    # assert the output
    # assert that the right elements are returned in 'covars_candidate_float_tensor'
    for i in range(covars_candidate_float_tensor.size()[1]):
        if isinstance(covars, list):
            assert covars_candidate_float_tensor[0, i].item() == covars[i]
        else:
            assert covars_candidate_float_tensor[0, i].item() == covars[0, i].item()


@pytest.mark.parametrize(
    "proposed_X",
    [torch.tensor([[1.1, 2.2]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
     None]
)
def test_get_and_verify_covars_input_fails(tmp_observe_class, proposed_X, monkeypatch):
    """
    test that _get_and_verify_covars_input fails for both providing incorrect data. Monkeypatching methods
    "_read_covars_manual_input" and "__validate_num_covars"
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    covariates = [1.1, 2.2, 200, -1.7]
    covars_tensor = torch.tensor([covariates], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # set proposed_X attribute (required for method to work)
    cls.proposed_X = proposed_X

    # monkeypatch "_read_covars_manual_input"
    def mock_read_covars_manual_input(x):
        return covars_tensor
    monkeypatch.setattr(cls, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_Validators__validate_num_covars"
    def mock_Validators__validate_num_covars(x):
        return False
    monkeypatch.setattr(cls, "_Validators__validate_num_covars", mock_Validators__validate_num_covars)

    # expected error message returned
    add_text = ""
    if cls.proposed_X is not None:
        add_text = " Was expecting something like '" + str(cls.proposed_X[-1]) + "', but got '" + str(covars_tensor) + "'"
    error_msg = "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations." + add_text

    # covariate kwargs is set to None so input-based method is used
    kwarg_covariates = None

    # run method
    with pytest.raises(Exception) as e:
        covars_candidate_float_tensor = cls._get_and_verify_covars_input(covars=kwarg_covariates)
    assert str(e.value) == error_msg


# negative tests for _get_and_verify_covars for kwargs input
@pytest.mark.parametrize(
    "covars, error_msg",
    [
        [[1.1, 2.2, 200, -1.7], "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations."],
        [torch.tensor([[1.1, 2.2, 200, -1.7]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations."],
        [torch.tensor([1.1, 2.2, 200, -1.7], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), "creative_project.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'. Was expecting torch tensor of size (1,<num_covariates>) but received one of size [4]"],  # this one fails in utils.__get_covars_from_kwargs because of wrong size of input tensor
    ]

)
def test_get_and_verify_covars_programmatic_fails(tmp_observe_class, covars, error_msg, monkeypatch):
    """
    test that _get_and_verify_covars_input fails when providing incorrect data programmatically. Monkeypatching
    method "__validate_num_covars". Expected error is related to wrong number of elements returned
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # temp class to execute the test
    cls = tmp_observe_class

    # monkeypatch "__get_covars_from_kwargs"
    def mock__get_covars_from_kwargs(x):
        if isinstance(covars, list):
            return torch.tensor([covars], dtype=torch.double, device=device)
        else:
            return covars
    monkeypatch.setattr(creative_project.utils, "__get_covars_from_kwargs", mock__get_covars_from_kwargs)

    # monkeypatch "_Validators__validate_num_covars"
    def mock_Validators__validate_num_covars(x):
        return False
    monkeypatch.setattr(cls, "_Validators__validate_num_covars", mock_Validators__validate_num_covars)

    # run method
    with pytest.raises(Exception) as e:
        covars_candidate_float_tensor = cls._get_and_verify_covars_input(covars=covars)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "train_X, covars_proposed_iter, covars_sampled_iter, kwarg_covariates",
    [
        [torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), 2, 1, None],
        [torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), 2, 1, torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))],
        [torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), 1, 1, None],
        [torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), 0, 0, None],
    ]
)
def test_covars_datapoint_observation_unit(tmp_observe_class, train_X, covars_proposed_iter, covars_sampled_iter, kwarg_covariates, monkeypatch):
    """
    test that _get_covars_datapoint works. Monkeypatching method "_get_and_verify_covars_input". Also test that this
    works both when covars is provided as kwargs or not (when covarites kwarg is set to None, different mehtod is used
    in _get_and_verify_covars_input; since we're monkeypatching anyways it shouldn't change, but testing anyways).
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    covariates = [1.1, 2.2, 200, -1.7]
    covars_tensor = torch.tensor([covariates], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # set proposed_X attribute (required for method to work)
    cls.proposed_X = train_X
    cls.train_X = train_X
    cls.model = {"covars_proposed_iter": covars_proposed_iter,
                 "covars_sampled_iter": covars_sampled_iter}

    # monkeypatch "_get_and_verify_covars_input"
    def mock_get_and_verify_covars_input(covars):
        return covars_tensor
    monkeypatch.setattr(cls, "_get_and_verify_covars_input", mock_get_and_verify_covars_input)

    # run the method being tested
    cls._get_covars_datapoint(covars=kwarg_covariates)

    # assert the right elements have been added
    for i in range(cls.train_X.size()[1]):
        assert cls.train_X[-1, i].item() == covariates[i]

    # assert that counter has been updated
    assert cls.model["covars_sampled_iter"] == cls.model["covars_proposed_iter"]

    # only if covars_proposed_iter is ahead of sampled
    if covars_proposed_iter > covars_sampled_iter:
        # assert that new row has been added
        assert cls.train_X.size()[0] == train_X.size()[0] + 1
    elif train_X is None:
        # assert that cls.train_X has been initiated
        assert cls.train_X.size()[0] == 1
    else:
        # assert that no new row has been added
        assert cls.train_X.size()[0] == train_X.size()[0]


@pytest.mark.parametrize(
    "train_Y, covars_proposed_iter, response_sampled_iter, kwarg_response",
    [
        [torch.tensor([[0.2]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), 2, 1, None],
        [torch.tensor([[0.2]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), 2, 1, [0.2]],
        [torch.tensor([[0.2]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), 2, 1, torch.tensor([[0.2]], dtype=torch.double)],
        [torch.tensor([[0.2]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), 1, 1, None],
        [None, 0, 0, None]
    ]
)
def test_response_datapoint_observation_unit(tmp_observe_class, train_Y, covars_proposed_iter, response_sampled_iter, kwarg_response, monkeypatch):
    """
    test that _get_response_datapoint works. Monkeypatching method "_get_and_verify_response_input". For iterative
    sampling,tests that it works both when response is provided as kwargs and not
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    resp = [1.1]
    resp_tensor = torch.tensor([resp], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # adding attributes required for test to work
    cls.train_Y = train_Y
    cls.model = {"covars_proposed_iter": covars_proposed_iter,
                 "response_sampled_iter": response_sampled_iter}

    # monkeypatch "_get_and_verify_covars_input"
    def mock_get_and_verify_response_input(response):
        return resp_tensor
    monkeypatch.setattr(cls, "_get_and_verify_response_input", mock_get_and_verify_response_input)

    # run the method being tested
    cls._get_response_datapoint(response=kwarg_response)

    # assert the right element have been added
    assert cls.train_Y[-1].item() == resp[0]

    # assert that counter has been updated
    assert cls.model["response_sampled_iter"] == cls.model["covars_proposed_iter"]

    # only if covars_proposed_iter is ahead of sampled
    if covars_proposed_iter > response_sampled_iter:
        # assert that new row has been added
        assert cls.train_Y.size()[0] == train_Y.size()[0] + 1
    elif train_Y is None:
        # assert that cls.train_X has been initiated
        assert cls.train_Y.size()[0] == 1
    else:
        # assert that no new row has been added
        assert cls.train_Y.size()[0] == train_Y.size()[0]