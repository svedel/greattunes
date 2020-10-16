from creative_project._initializers import Initializers
import torch


def test_Initializers__initialize_best_response_functional(custom_models_simple_training_data_4elements):
    """
    test initialization of best response data structures based on input data
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # initialize class and register required attributes
    cls = Initializers()
    cls.train_X = train_X
    cls.train_Y = train_Y

    print(cls.train_Y)

    # run the function
    cls._Initializers__initialize_best_response()

    # assert that the function has been applied
    assert isinstance(cls.covars_best_response_value, torch.Tensor)
    assert isinstance(cls.best_response_value, torch.Tensor)

    # check size
    assert cls.covars_best_response_value.shape[0] == train_X.shape[0]
    assert cls.covars_best_response_value.shape[1] == train_X.shape[1]
    assert cls.best_response_value.shape[0] == train_Y.shape[0]
    assert cls.best_response_value.shape[1] == train_Y.shape[1]

    # check values, compare to mock_find_max_response_value
    for it in range(train_X.shape[0]):
        assert cls.covars_best_response_value[it].item() == torch.tensor([[1.0]], dtype=torch.double).item() & cls.best_response_value[it].item() == torch.tensor([[1.0]], dtype=torch.double).item()