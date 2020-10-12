"""
This file contains a helper class with initializer methods. It is only wrapped as a class for convenience. Will
therefore only test individual functions
"""

import pytest
import torch
from creative_project._initializers import Initializers


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