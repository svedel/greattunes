import numpy as np
import pytest
import torch
from creative_project._validators import Validators
from creative_project._initializers import Initializers


@pytest.mark.parametrize("dataset_id",[0, 1])
def test_Validators__validate_num_covars(covars_initialization_data, dataset_id):
    """
    test that this works by providing data of correct and incorrect sizes
    """

    covars = covars_initialization_data[dataset_id]
    covars_array = torch.tensor([[g[0]] for g in covars], dtype=torch.double)
    covars_array_wrong = torch.cat((covars_array, torch.tensor([[22.0]], dtype=torch.double)))

    # instatiate Validators class
    cls = Validators()

    # Leverage functionality from creative_project._initializers.Initializers.__initialize_from_covars for setting
    # correct data for test of __validate_num_covars
    init_cls = Initializers()
    init_cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_cls.dtype = torch.double

    cls.initial_guess, _ = init_cls._Initializers__initialize_from_covars(covars=covars)

    # test Validator.__validate_num_covars
    assert cls._Validators__validate_num_covars(covars_array)
    assert not cls._Validators__validate_num_covars(covars_array_wrong)


def test_unit_Validators__validate_training_data(custom_models_simple_training_data_4elements, monkeypatch):
    """
    unit test __validate_training_data. Use monkeypatching for embedded call to Validators.__validate_num_covars.
    """

    # initialize the class
    cls = Validators()

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # monkeypatching private method in order to keep to unit test
    def mock__validate_num_covars(train_X):
        return True
    monkeypatch.setattr(
        cls, "_Validators__validate_num_covars", mock__validate_num_covars
    )

    # run the test with OK input
    assert cls._Validators__validate_training_data(train_X=train_X, train_Y=train_Y)

    # set one input dataset to None, will fail
    assert not cls._Validators__validate_training_data(train_X=train_X, train_Y=None)

    # change datatype to numpy for train_Y, will fail
    assert not cls._Validators__validate_training_data(train_X=train_X, train_Y=train_Y.numpy())

    # change number of entries to train_Y, will fail
    new_train_Y = torch.cat((train_Y, torch.tensor([[22.0]], dtype=torch.double)))
    assert not cls._Validators__validate_training_data(train_X=train_X, train_Y=new_train_Y)


@pytest.mark.parametrize(
    "covars, error_msg",
    [
        [None, "kre8_core.creative_project._validators.Validator.__validate_covars: covars is None"],  # checks no covars data provided
        [(1, 2, 3), "kre8_core.creative_project._validators.Validator.__validate_covars: covars is not list of tuples (not list)"],  # checks fail if covars not a list of tuples
        [[1, 2, 3], "kre8_core.creative_project._validators.Validator.__validate_covars: entry in covars list is not tuple"],  # checks that covars is a list of tuples
        [[("hej", 2, 3)], "kre8_core.creative_project._validators.Validator.__validate_covars: tuple element hej in covars list is neither of type float or int"]  # test that all elements in tuples are of type int or float
    ])
def test_Validators__validate_covars_exceptions(covars, error_msg):
    """
    test conditions under which private method __validate_covars fails, check exceptions raised
    """

    cls = Validators()

    with pytest.raises(Exception) as e:
        # set up class
        assert cls._Validators__validate_covars(covars=covars)
    assert str(e.value) == error_msg


def test_Validators__validate_covars_passes(covars_for_custom_models_simple_training_data_4elements):
    """
    test conditions under which private method __validate_covars passes
    """

    covars = covars_for_custom_models_simple_training_data_4elements

    cls = Validators()

    assert cls._Validators__validate_covars(covars=covars)