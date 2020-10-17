"""
This file contains a helper class with initializer methods. It is only wrapped as a class for convenience. Will
therefore only test individual functions
"""
import pytest
import torch
from creative_project._initializers import Initializers
import creative_project._max_response

@pytest.mark.parametrize("dataset_id",[0, 1])
def test_Initializers__initialize_from_covars(covars_initialization_data, dataset_id):
    """
    test that simple data input data is processed and right class attributes set
    """

    # input data
    covars = covars_initialization_data[dataset_id]
    num_cols = len(covars)

    # initialize class
    cls = Initializers()

    # define new class attributes from torch required for method to run (method assumes these defined under
    # main class in creative_project.__init__.py)
    cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls.dtype = torch.double

    # run method
    initial_guesses, bounds = cls._Initializers__initialize_from_covars(covars=covars)

    # asserts type
    assert isinstance(initial_guesses, torch.Tensor)
    assert isinstance(bounds, torch.Tensor)

    shape_initial_guesses = [x for x in initial_guesses.shape]
    assert shape_initial_guesses[0] == 1
    assert shape_initial_guesses[1] == num_cols

    shape_bounds = [x for x in bounds.shape]
    assert shape_bounds[0] == 2
    assert shape_bounds[1] == num_cols


def test_Initializers__initialize_best_response_first_add(custom_models_simple_training_data_4elements, monkeypatch):
    """
    test initialization of best response data structures based on input data
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # monkeypatch function call
    tmp_output = 1.0
    def mock_find_max_response_value(train_X, train_Y):
        return torch.tensor([[tmp_output]], dtype=torch.double), torch.tensor([[tmp_output]], dtype=torch.double)
    monkeypatch.syspath_prepend("..")
    monkeypatch.setattr(
        creative_project._max_response, "find_max_response_value", mock_find_max_response_value
    )

    # initialize class and register required attributes
    cls = Initializers()
    cls.train_X = train_X
    cls.train_Y = train_Y

    # run the functio(
    cls._Initializers__initialize_best_response()

    print("returned covars")
    print(cls.covars_best_response_value)
    print("returned resp")
    print(cls.best_response_value)

    # assert that the function has been applied
    assert isinstance(cls.covars_best_response_value, torch.Tensor)
    assert isinstance(cls.best_response_value, torch.Tensor)

    # check size
    assert cls.covars_best_response_value.shape[0] == train_X.shape[0]
    assert cls.best_response_value.shape[0] == train_Y.shape[0]

    # check values, compare to mock_find_max_response_value
    for it in range(train_X.shape[0]):
        assert cls.covars_best_response_value[it].item() == tmp_output
        assert cls.best_response_value[it].item() == tmp_output