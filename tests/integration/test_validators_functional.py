import pytest
import torch
from greattunes._validators import Validators
from greattunes._initializers import Initializers


def test_functional_Validators__validate_training_data(custom_models_simple_training_data_4elements, covars_for_custom_models_simple_training_data_4elements):
    """
    functional test of private method __validate_training_data, relying on all embedded function calls as-is
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]
    #covars = [(x[0], x[0]-1.0, x[0]+1.0) for x in train_X.numpy()]  # create covars in right format from train_X
    covars = covars_for_custom_models_simple_training_data_4elements

    # initialize the class
    cls = Validators()
    cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # adding a few attributes required for test
    cls.dtype = torch.double

    # add initial guess to cls, which is required for this to work
    init_cls = Initializers()
    init_cls.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # adding a few attributes required for test
    init_cls.dtype = torch.double
    cls.initial_guess, _ = init_cls._Initializers__initialize_from_covars(covars=covars)


    # run the test with OK input
    assert cls._Validators__validate_training_data(train_X=train_X, train_Y=train_Y)

    # set one input dataset to None, will fail
    assert not cls._Validators__validate_training_data(train_X=train_X, train_Y=None)

    # change datatype to numpy for train_Y, will fail
    assert not cls._Validators__validate_training_data(train_X=train_X, train_Y=train_Y.numpy())

    # change number of entries to train_Y, will throw exception
    new_train_Y = torch.cat((train_Y, torch.tensor([[22.0]], dtype=torch.double)))
    with pytest.raises(Exception) as e:
        #assert not cls._Validators__validate_training_data(train_X=train_X, train_Y=new_train_Y)
        cls._Validators__validate_training_data(train_X=train_X, train_Y=new_train_Y)
    assert str(e.value) == "greattunes._validators.Validators.__validate_training_data: The number of rows (observations) in provided 'train_X' (4) is not the same as for `train_Y` (5) as it should be."
