import pytest


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