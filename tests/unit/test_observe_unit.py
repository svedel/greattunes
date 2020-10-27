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
        assert output[0,it].item() == input_data[it]

