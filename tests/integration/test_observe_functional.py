import pandas as pd
import pytest
import torch


@pytest.mark.parametrize("input_data", [[4.5],])
def test_observe_get_and_verify_response_input_manual_functional(tmp_observe_class, input_data, monkeypatch):
    """
    test _get_and_verify_response_input for manual method for getting response values. Monkeypatching the built-in
    "input" function
    """

    # initialized temp class
    cls = tmp_observe_class
    cls.sampling["method"] = "iterative"

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


@pytest.mark.parametrize(
    "input_data, error_msg",
    [
        [[3.1, -12.2], "creative_project._observe._get_and_verify_response_input: incorrect number of variables provided. Was expecting input of size (1,1) but received torch.Size([1, 2])"],
        ]
)
def test_observe_get_and_verify_response_input_manual_functional_fails(tmp_observe_class, input_data, error_msg, monkeypatch):
    """
    test _get_and_verify_response_input for manual method for getting response values. Monkeypatching the built-in
    "input" function
    """

    # initialized temp class
    cls = tmp_observe_class
    cls.sampling["method"] = "iterative"

    # set attribute
    cls.model = {"covars_proposed_iter": 0}

    # monkeypatching "input"
    monkeypatch_output = ", ".join([str(x) for x in input_data])  # match data from "input" function
    monkeypatch.setattr("builtins.input", lambda _: monkeypatch_output)

    # run function
    with pytest.raises(Exception) as e:
        output = cls._get_and_verify_response_input()
    assert str(e.value) == error_msg


def test_observe_get_and_verify_response_input_functions_functional(tmp_observe_class, training_data_covar_complex):
    """
    test _get_and_verify_response_input for manual method for getting response values
    """

    # initialized temp class
    cls = tmp_observe_class
    cls.sampling["method"] = "functions"
    cls.train_X = training_data_covar_complex[1]
    cls.covar_details = training_data_covar_complex[3]

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

    # set kwarg response to None (so manually provided input is used)
    kwarg_response = None

    # run test
    output = cls._get_and_verify_response_input()

    assert output[0].item() == tmp_val


@pytest.mark.parametrize(
    "covariates, kwarg_covariates",
    [
        [[1.1, 2.2, 200, -1.7], None],
        [[1.1, 2.2, 200, -1.7], [1.1, 2.2, 200, -1.7]],
    ]
)
def test_get_and_verify_covars_input_with_dependencies_works(tmp_observe_class,
                                                             covar_details_mapped_covar_mapped_names_tmp_observe_class,
                                                             covariates, kwarg_covariates, monkeypatch):
    """
    test _get_and_verify_covars_input which depends on _read_covars_manual_input and
    _validators.Validators.__validate_num_covars. monkeypatch the user input (when needed), also test the programmatic
    option to provide input via kwargs
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    covars_tensor = torch.tensor([covariates], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # define required attributes
    cls.proposed_X = covars_tensor
    cls.initial_guess = covars_tensor
    cls.covar_details = covar_details_mapped_covar_mapped_names_tmp_observe_class[0]
    cls.covar_mapped_names = covar_details_mapped_covar_mapped_names_tmp_observe_class[1]
    cls.sorted_pandas_columns = covar_details_mapped_covar_mapped_names_tmp_observe_class[2]

    # monkeypatch
    def mock_input(x):  # mock function to replace 'input' for unit testing purposes
        return ", ".join([str(x) for x in covariates])
    monkeypatch.setattr("builtins.input", mock_input)

    # run method
    covars_candidate_float_tensor = cls._get_and_verify_covars_input(covars=kwarg_covariates)

    # assert
    for i in range(covars_candidate_float_tensor.size()[1]):
        assert covars_candidate_float_tensor[0, i].item() == covariates[i]


@pytest.mark.parametrize(
    "covariates, kwarg_covariates",
    [
        [[1.1, 2.2, 200, -1.7], None],
        [[1.1, 2.2, 200, -1.7], [1.1, 2.2, 200, -1.7]],
    ]
)
def test_get_and_verify_covars_input_with_dependencies_fails(tmp_observe_class, covariates, kwarg_covariates, monkeypatch):
    """
    test _get_and_verify_covars_input which depends on _read_covars_manual_input and
    _validators.Validators.__validate_num_covars. monkeypatch the user input (when needed), also test the programmatic
    option to provide input via kwargs
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
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
    error_msg = "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like '" + str(cls.proposed_X[-1]) + "', but got '" + str(covars_tensor) + "'"

    # run the method, expect it to fail
    with pytest.raises(Exception) as e:
        covars_candidate_float_tensor = cls._get_and_verify_covars_input(covars=kwarg_covariates)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "train_X, x_data, covars_proposed_iter, covars_sampled_iter, covariates, kwarg_covariates",
    [
        [torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({"covar0": [0.1], "covar1": [2.5], "covar2": [12], "covar3": [0.22]}), 2, 1, [1.1, 2.2, 200, -1.7], [1.1, 2.2, 200, -1.7]],
        [torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({"covar0": [0.1], "covar1": [2.5], "covar2": [12], "covar3": [0.22]}), 2, 1, [1.1, 2.2, 200, -1.7], None],
        [torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({"covar0": [0.1], "covar1": [2.5], "covar2": [12], "covar3": [0.22]}), 2, 1, [1.1, 2.2, 200, -1.7], torch.tensor([[1.1, 2.2, 200, -1.7]], dtype=torch.double)],
        [torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({"covar0": [0.1], "covar1": [2.5], "covar2": [12], "covar3": [0.22]}), 1, 1, [1.1, 2.2, 200, -1.7], None],
        #[torch.tensor([[0.1, 2.5, 12, 0.22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({"covar0": [0.1], "covar1": [2.5], "covar2": [12], "covar3": [0.22]}), 0, 0, [1.1, 2.2, 200, -1.7], None],
        [None, None, 0, 0, [1.1, 2.2, 200, -1.7], None],
    ]
)
def test_covars_datapoint_observation_int_works(tmp_observe_class,
                                                covar_details_mapped_covar_mapped_names_tmp_observe_class,
                                                train_X, x_data, covars_proposed_iter, covars_sampled_iter, covariates,
                                                kwarg_covariates, monkeypatch):
    """
    test that _get_covars_datapoint works. Monkeypatching build-in method "input"
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    covars_tensor = torch.tensor([covariates], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # set proposed_X attribute (required for method to work)
    cls.initial_guess = covars_tensor
    cls.proposed_X = covars_tensor
    cls.train_X = train_X
    cls.x_data = x_data
    cls.model = {"covars_proposed_iter": covars_proposed_iter,
                 "covars_sampled_iter": covars_sampled_iter}
    cls.covar_details = covar_details_mapped_covar_mapped_names_tmp_observe_class[0]
    cls.covar_mapped_names = covar_details_mapped_covar_mapped_names_tmp_observe_class[1]
    cls.sorted_pandas_columns = covar_details_mapped_covar_mapped_names_tmp_observe_class[2]

    # monkeypatch
    def mock_input(x):  # mock function to replace 'input' for unit testing purposes
        return ", ".join([str(x) for x in covariates])
    monkeypatch.setattr("builtins.input", mock_input)

    # run the method
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
    "covariate_str, error_msg",
    [
        #["", "could not convert string to float: ''"],
        ["", "creative_project._observe._read_covars_manual_input: incorrect number of covariates (1) provided, was expecting 4"],
        ["1, a, 2, 3", "could not convert string to float: ' a'"],
        [" , a, 2, 3", "could not convert string to float: ''"]
    ]
)
def test_covars_datapoint_observation_int_fails(tmp_observe_class,
                                                covar_details_mapped_covar_mapped_names_tmp_observe_class,
                                                covariate_str, error_msg, monkeypatch, pythontestvers):
    """
    test that _get_covars_datapoint fails. Monkeypatching build-in method "input"
    """

    # special case for python version 3.7 (handled via new keyword argument to pytest)
    if pythontestvers == "3.7" and covariate_str != "1, a, 2, 3":
        # removes the '' from the error message
        error_msg = error_msg[:-2]

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    init_guess = [1.1, 2.2, 200, -1.7]
    initial_guess = torch.tensor([init_guess], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # set proposed_X attribute (required for method to work)
    cls.initial_guess = initial_guess
    cls.proposed_X = initial_guess
    cls.train_X = initial_guess
    cls.model = {"covars_proposed_iter": 1,
                 "covars_sampled_iter": 1}
    cls.covar_details = covar_details_mapped_covar_mapped_names_tmp_observe_class[0]
    cls.covar_mapped_names = covar_details_mapped_covar_mapped_names_tmp_observe_class[1]
    cls.sorted_pandas_columns = covar_details_mapped_covar_mapped_names_tmp_observe_class[2]
    cls.x_data = pd.DataFrame(columns=["covar0", "covar1", "covar2", "covar3"])
    cls.x_data.loc[0] = init_guess

    # monkeypatch
    def mock_input(x):  # mock function to replace 'input' for unit testing purposes
        return covariate_str
    monkeypatch.setattr("builtins.input", mock_input)

    # covariate kwargs is set to None so input-based method is used
    kwarg_covariates = None

    #with pytest.raises(ValueError) as e:
    with pytest.raises(Exception) as e:
        # run the method
        cls._get_covars_datapoint(covars=kwarg_covariates)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "kwarg_covariates, error_msg",
    [
        [[1.1, 2.2, -1.7], "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like 'tensor([  1.1000,   2.2000, 200.0000,  -1.7000], dtype=torch.float64)', but got 'tensor([[ 1.1000,  2.2000, -1.7000]], dtype=torch.float64)'"],
        [torch.tensor([1.1, 2.2, 200, -1.7], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), "creative_project.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'. Was expecting torch tensor of size (1,<num_covariates>) but received one of size (4)."]
    ]
)
def test_get_covars_datapoint_kwargs_data_int_fails(tmp_observe_class,
                                                    covar_details_mapped_covar_mapped_names_tmp_observe_class,
                                                    kwarg_covariates, error_msg):
    """
    test that _get_covars_datapoint fails when providing incorrect kwargs
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    init_guess = [1.1, 2.2, 200, -1.7]
    initial_guess = torch.tensor([init_guess], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # set proposed_X attribute (required for method to work)
    cls.initial_guess = initial_guess
    cls.proposed_X = initial_guess
    cls.train_X = initial_guess
    cls.model = {"covars_proposed_iter": 1,
                 "covars_sampled_iter": 1}
    cls.covar_details = covar_details_mapped_covar_mapped_names_tmp_observe_class[0]
    cls.covar_mapped_names = covar_details_mapped_covar_mapped_names_tmp_observe_class[1]
    cls.sorted_pandas_columns = covar_details_mapped_covar_mapped_names_tmp_observe_class[2]

    with pytest.raises(Exception) as e:
        # run the method
        cls._get_covars_datapoint(covars=kwarg_covariates)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "train_Y, y_data, covars_proposed_iter, response_sampled_iter",
    [
        [torch.tensor([[0.1]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({"Response": [0.1]}), 2, 1],
        [torch.tensor([[0.1]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({"Response": [0.1]}), 1, 1],
        #[torch.tensor([[0.1]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), pd.DataFrame({"Response": [0.1]}), 0, 0],
        [None, None, 0, 0],
    ]
)
def test_response_datapoint_observation_works(tmp_observe_class, train_Y, y_data, covars_proposed_iter, response_sampled_iter, monkeypatch):
    """
    test that _get_response_datapoint works. Monkeypatching build-in method "input"
    """

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    resp = [2.2]
    resp_tensor = torch.tensor([resp], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # set proposed_X attribute (required for method to work)
    cls.initial_guess = torch.tensor([[1, 2, 3]], dtype=torch.double, device=device)#resp_tensor
    cls.proposed_X = torch.tensor([[1, 2, 3]], dtype=torch.double, device=device)#resp_tensor
    cls.train_Y = train_Y
    cls.model = {"covars_proposed_iter": covars_proposed_iter,
                 "response_sampled_iter": response_sampled_iter}
    cls.y_data = y_data

    # monkeypatch
    def mock_input(x):  # mock function to replace 'input' for unit testing purposes
        return ", ".join([str(x) for x in resp])
    monkeypatch.setattr("builtins.input", mock_input)

    # set kwarg response to None (so manually provided input is used)
    kwarg_response = None

    # run the method being tested
    cls._get_response_datapoint(response=kwarg_response)

    # assert the right elements have been added
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


@pytest.mark.parametrize(
    "response_str, kwarg_response, error_msg",
    [
        ["", None, "could not convert string to float: ''"],
        ["a", None, "could not convert string to float: 'a'"],
        [" , a", None, "could not convert string to float: ''"],
        ["1", [1, 2], "creative_project._observe._get_and_verify_response_input: incorrect number of variables provided. Was expecting input of size (1,1) but received torch.Size([1, 2])"],
        ["1", ['a'], "too many dimensions 'str'"],
        ["1", torch.tensor([[1, 2]], dtype=torch.double), "creative_project.utils.__get_response_from_kwargs: dimension mismatch in provided 'response'. Was expecting torch tensor of size (1,1) but received one of size (1, 2)."]
    ]
)
def test_response_datapoint_observation_fails(tmp_observe_class, response_str, kwarg_response, error_msg, monkeypatch, pythontestvers):
    """
    test that _get_response_datapoint fails under right conditions. Monkeypatching build-in method "input" when testing
    this method for providing response input, but also tests failure of programmatically providing input using the
    kwarg "response" (default mode: if "response" is not None, it will be used and no manual data will be needed)
    """

    # special case for python version 3.7 (handled via new keyword argument to pytest)
    if pythontestvers == "3.7" and (response_str == "" or response_str == " , a"):
        # removes the '' from the error message
        error_msg = error_msg[:-2]

    # device for torch tensor definitions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # covariates to sample
    resp = [2.2]
    resp_tensor = torch.tensor([resp], dtype=torch.double, device=device)

    # temp class to execute the test
    cls = tmp_observe_class

    # set proposed_X attribute (required for method to work)
    cls.sampling = {"method": "iterative"}
    cls.initial_guess = resp_tensor
    cls.proposed_X = resp_tensor
    cls.train_Y = resp_tensor
    cls.model = {"covars_proposed_iter": 1,
                 "response_sampled_iter": 1}

    # monkeypatch
    def mock_input(x):  # mock function to replace 'input' for unit testing purposes
        return response_str
    monkeypatch.setattr("builtins.input", mock_input)

    with pytest.raises(Exception) as e:
        # run the method
        cls._get_response_datapoint(response=kwarg_response)
    assert str(e.value) == error_msg
