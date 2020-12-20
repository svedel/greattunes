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


@pytest.mark.parametrize(
    "response_sampled_iter, covars_best, response_best",
    [
        [3, [1.1], [2.2]],
        [10, [3.5, 12.2], [1.7]]
    ]
)
def test_current_best_univariate_unit(tmp_best_response_class, response_sampled_iter, covars_best, response_best):
    """
    test that current_best works for univariate and multivariate covariate data
    """

    # class used for test
    cls = tmp_best_response_class

    # set attribute
    cls.model = {"response_sampled_iter": response_sampled_iter}

    # set best response variables
    cls.covars_best_response_value = torch.tensor([covars_best], dtype=torch.double)
    cls.best_response_value = torch.tensor([response_best], dtype=torch.double)

    # run method
    cls.current_best()

    # assert
    for it in range(len(covars_best)):
        assert cls.best["covars"][it] == covars_best[it]
    assert cls.best["response"] == response_best[0]
    assert cls.best["iteration_when_recorded"] == response_sampled_iter


@pytest.mark.parametrize(
    "candidate, proposed_X, covars_sampled_iter",
    [
        [torch.tensor([[1]], dtype=torch.double), torch.tensor([[0.9]], dtype=torch.double), 1],
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.9, 1.1, 200]], dtype=torch.double), 1],
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.9, 1.1, 200], [0, 1, 2]], dtype=torch.double), 2]
    ]
)
def test_update_proposed_data_works(tmp_best_response_class, candidate, proposed_X, covars_sampled_iter):
    """
    positive tests for the cases where "_update_proposed_data" works
    :param tmp_best_response_class:
    :param candidate:
    :param proposed_X:
    :param covars_sampled_iter:
    :return:
    """

    # initialize temp class for running test
    cls = tmp_best_response_class

    # update attributes to run test
    cls.model = {
        "covars_sampled_iter": covars_sampled_iter,
        "covars_proposed_iter": 0,  # can initialize at any value, is updated relative to "covars_sampled_iter"
    }
    cls.proposed_X = proposed_X

    # run the test
    cls._update_proposed_data(candidate=candidate)

    # assert that cls.proposed_X has grown by one row
    assert cls.proposed_X.size()[0] == proposed_X.size()[0] + 1

    # assert content of that last row of cls.proposed_X
    for i in range(cls.proposed_X.size()[1]):
        assert cls.proposed_X[-1, i].item() == candidate[0, i].item()
