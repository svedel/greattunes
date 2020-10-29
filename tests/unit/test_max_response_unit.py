import torch
import pytest
from creative_project._best_response import _find_max_response_value, _update_max_response_value


def test_find_max_response_value_unit(custom_models_simple_training_data_4elements,
                                      tmp_best_response_class):
    """
    test finding of max response value
    """

    # data -- max at 4th element
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # test class for all methods from creative_project._best_response --- only an infrastructure ease for
    # _find_max_reponse_value since it's a static method
    cls = tmp_best_response_class

    max_X, max_Y = cls._find_max_response_value(train_X, train_Y)

    # assert that max value is at index 3 (4th element)
    maxit = 3
    assert max_X[0].item() == train_X[maxit].item()
    assert max_Y[0].item() == train_Y[maxit].item()


def test_find_max_response_value_multivariate(training_data_covar_complex, tmp_best_response_class):
    """
    test that the right covariate row tensor is returned when train_X is multivariate
    """

    # data
    train_X = training_data_covar_complex[1]
    train_Y = training_data_covar_complex[2]

    # test class for all methods from creative_project._best_response --- only an infrastructure ease for
    # _find_max_reponse_value since it's a static method
    cls = tmp_best_response_class

    max_X, max_Y = cls._find_max_response_value(train_X, train_Y)

    # assert that max value is at index 1 (2nd element)
    maxit = 1
    for it in range(train_X.shape[1]):
        assert max_X[0,it].item() == train_X[maxit, it].item()
    assert max_Y[0].item() == train_Y[maxit].item()


def test_update_max_response_value_unit(custom_models_simple_training_data_4elements, tmp_best_response_class,
                                        monkeypatch):
    """
    test that _update_max_response_value works for univariate data
    """

    # data -- max at 4th element
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # test class
    cls = tmp_best_response_class

    # set some attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y

    # monkeypatch
    max_X = 1.0
    max_Y = 2.0
    def mock_find_max_response_value(train_X, train_Y):
        mX = torch.tensor([max_X], dtype=torch.double,
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        mY = torch.tensor([max_Y], dtype=torch.double,
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return mX, mY
    monkeypatch.setattr(
        cls, "_find_max_response_value", mock_find_max_response_value
    )

    # test that it initializes storage of best results when none present
    cls._update_max_response_value()

    assert cls.covars_best_response_value[0].item() == max_X
    assert cls.best_response_value[0].item() == max_Y
