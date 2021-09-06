from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
import torch
import pytest
from greattunes._best_response import _find_max_response_value, _update_max_response_value


def test_find_max_response_value_unit(custom_models_simple_training_data_4elements,
                                      tmp_best_response_class):
    """
    test finding of max response value
    """

    # data -- max at 4th element
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # test class for all methods from greattunes._best_response --- only an infrastructure ease for
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

    # test class for all methods from greattunes._best_response --- only an infrastructure ease for
    # _find_max_reponse_value since it's a static method
    cls = tmp_best_response_class

    max_X, max_Y = cls._find_max_response_value(train_X, train_Y)

    # assert that max value is at index 1 (2nd element)
    maxit = 1
    for it in range(train_X.shape[1]):
        assert max_X[0,it].item() == train_X[maxit, it].item()
    assert max_Y[0].item() == train_Y[maxit].item()


def test_update_max_response_value_unit(custom_models_simple_training_data_4elements, tmp_best_response_class,
                                        custom_models_simple_training_data_4elements_covar_details, monkeypatch):
    """
    test that _update_max_response_value works for univariate data
    """

    # data -- max at 4th element
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]
    covar_details = custom_models_simple_training_data_4elements_covar_details[0]
    covar_mapped_names = custom_models_simple_training_data_4elements_covar_details[1]

    # test class
    cls = tmp_best_response_class

    # set some attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y
    cls.covar_details = covar_details
    cls.covar_mapped_names = covar_mapped_names

    # monkeypatch
    max_X = 1.0
    max_Y = 2.0
    def mock_find_max_response_value(train_X, train_Y):
        mX = torch.tensor([[max_X]], dtype=torch.double,
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        mY = torch.tensor([[max_Y]], dtype=torch.double,
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
    cls.covar_details = {}
    for i in range(len(covars_best)):
        tmp_key = "covar" + str(i)
        cls.covar_details[tmp_key] = {"guess": covars_best[i],
                                      "min": covars_best[i]-1,
                                      "max": covars_best[i]+1,
                                      "type": float,
                                      "columns": i,
                                      "pandas_column": i}
    cls.total_num_covars = len(covars_best)

    # set best response variables
    cls.covars_best_response_value = torch.tensor([covars_best], dtype=torch.double)
    cls.best_response_value = torch.tensor([response_best], dtype=torch.double)

    # run method
    cls.current_best()

    # assert
    for it in range(len(covars_best)):
        assert cls.best["covars"].values[0][it] == covars_best[it]
    assert cls.best["response"].values[0][0] == response_best[0]
    assert cls.best["iteration_when_recorded"] == response_sampled_iter


@pytest.mark.parametrize(
    "candidate, proposed_X, covars_sampled_iter",
    [
        [torch.tensor([[1]], dtype=torch.double), torch.tensor([[0.9]], dtype=torch.double), 1],  # univariate
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.9, 1.1, 200]], dtype=torch.double), 1],  # multivariate
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.9, 1.1, 200], [0, 1, 2]], dtype=torch.double), 2]  # multivariate and several historical entries
    ]
)
def test_update_proposed_data_works(tmp_best_response_class, candidate, proposed_X, covars_sampled_iter):
    """
    positive tests for the cases where "_update_proposed_data" works
    :param: tmp_best_response_class (test class defined in conftest.py)
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


@pytest.mark.parametrize(
    "candidate, proposed_X, covars_sampled_iter",
    [
        [torch.tensor([[1]], dtype=torch.double), torch.tensor([[0.9], [1.1]], dtype=torch.double), 1],
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.9, 1.1, 200], [1.1, 2.2, 4.3]], dtype=torch.double), 1],
        [torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([[0.9, 1.1, 200], [1.1, 2.2, 4.3], [0, 1, 2]], dtype=torch.double), 2]
    ]
)
def test_update_proposed_data_overwrite(tmp_best_response_class, candidate, proposed_X, covars_sampled_iter):
    """
    positive tests for the cases where "_update_proposed_data" overwrites last entry in self.proposed_X
    :param: tmp_best_response_class (test class defined in conftest.py)
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

    # assert that cls.proposed_X has NOT grown
    assert cls.proposed_X.size()[0] == proposed_X.size()[0]

    # assert content of that last row of cls.proposed_X
    for i in range(cls.proposed_X.size()[1]):
        assert cls.proposed_X[-1, i].item() == candidate[0, i].item()


@pytest.mark.parametrize(
    "candidate", [torch.tensor([[1]], dtype=torch.double), torch.tensor([[1, 2, 3]], dtype=torch.double)]
)
def test_update_proposed_data_works_firs_entry(tmp_best_response_class, candidate):
    """
    positive tests for the cases where "_update_proposed_data" stores first set of candidate datapoints
    :param: tmp_best_response_class (test class defined in conftest.py)
    """

    # initialize temp class for running test
    cls = tmp_best_response_class

    # update attributes to run test. self.proposed_X is initialized as 'None' in conftest.py
    cls.model = {
        "covars_sampled_iter": 0,  # initialize at 0 for the test to work (no data points sampled yet)
        "covars_proposed_iter": 1,  # initialize at 1 since one datapoint has been proposed (can actually initialize at any value for _update_proposed_data to work)
    }

    # run the test
    cls._update_proposed_data(candidate=candidate)

    # assert that only a single candidate datapoint has been added
    assert cls.proposed_X.size()[0] == 1

    # assert that it's the right datapoint by comparing to 'candidate'
    for i in range(cls.proposed_X.size()[1]):
        assert cls.proposed_X[-1, i].item() == candidate[0, i].item()


@pytest.mark.parametrize(
    "candidate, proposed_X, error_msg",
    [
        [None, torch.tensor([[0.9], [1.1]], dtype=torch.double), "greattunes._best_response._update_proposed_data: provided variable is not 'candidate' of type torch.DoubleTensor (type of 'candidate' is " + str(type(None)) + ")"],
        [[1.1], torch.tensor([[0.9], [1.1]], dtype=torch.double), "greattunes._best_response._update_proposed_data: provided variable is not 'candidate' of type torch.DoubleTensor (type of 'candidate' is " + str(type([1.1])) + ")"],
        [torch.tensor([[1, 2]], dtype=torch.double), torch.tensor([[0.9], [1.1]], dtype=torch.double), "greattunes._best_response._update_proposed_data: wrong number of covariates provided in 'candidate'. Expected 1, but got 2"],
        [torch.tensor([[1, 2]], dtype=torch.double), torch.tensor([[0.9, 1.1, 200], [1.1, 2.2, 4.3], [0, 1, 2]], dtype=torch.double), "greattunes._best_response._update_proposed_data: wrong number of covariates provided in 'candidate'. Expected 3, but got 2"]
    ]
)
def test_update_proposed_data_fails(tmp_best_response_class, candidate, proposed_X, error_msg):
    """
    negative tests for "_update_proposed_data": wrong data types, wrong number of candidates
    :param: tmp_best_response_class (test class defined in conftest.py)
    """

    # initialize temp class for running test
    cls = tmp_best_response_class

    # update attributes to run test. self.proposed_X is initialized as 'None' in conftest.py
    cls.model = {
        "covars_sampled_iter": 0,  # initialize at 0 for the test to work (no data points sampled yet)
        "covars_proposed_iter": 1,
        # initialize at 1 since one datapoint has been proposed (can actually initialize at any value for _update_proposed_data to work)
    }
    cls.proposed_X = proposed_X

    # run the test
    with pytest.raises(AssertionError) as e:
        cls._update_proposed_data(candidate=candidate)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "maximize, result",
    [
        [True, 1],
        [False, 0],
    ]
)
def test_find_best_predicted_1d(tmp_best_response_class, monkeypatch, maximize, result):
    """"
    tests that the optimizing method indeed finds the maximum. Monkeypatching the function evaluation step
    """

    # defines temporary class
    tmp = tmp_best_response_class

    # add some attributes to tmp
    tmp.dtype = torch.double
    tmp.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # add function to maximize
    def f(type, x):
        return torch.sin(x**2)

    monkeypatch.setattr(tmp, "_evaluate_model", f)

    # initialize x and type (ignore type in this test)
    #x_init = torch.rand(1).requires_grad_(True)
    x_init = torch.rand(1, 1, dtype=torch.double)
    type = "mean"

    # run the method
    y, x = tmp._find_best_predicted(x_init, type, maximize=maximize, max_iter=20)

    # assert
    assert abs(y-result) < 1e-3


@pytest.mark.parametrize(
    "maximize, result",
    [
        [True, 1],
        [False, 0],
    ]
)
def test_find_best_predicted_2d(tmp_best_response_class, monkeypatch, maximize, result):
    """"
    tests that the optimizing method indeed finds the maximum. Monkeypatching the function evaluation step
    """

    # defines temporary class
    tmp = tmp_best_response_class

    # add some attributes to tmp
    tmp.dtype = torch.double
    tmp.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # add function to maximize
    def f(type, x):
        return (torch.sin(x[:,0]**2)*torch.sin(x[:,1]**2)).unsqueeze(1)

    monkeypatch.setattr(tmp, "_evaluate_model", f)

    # initialize x and type (ignore type in this test)
    #x_init = torch.rand(1, 2).requires_grad_(True)
    x_init = torch.rand(1, 2, dtype=torch.double)
    type = "mean"

    # run the method
    y,x = tmp._find_best_predicted(x_init, type, maximize=maximize, max_iter=20)

    # assert
    assert abs(y-result) < 1e-3


@pytest.mark.parametrize(
    "type, result",
    [
        ["mean", 1.2500295534805428],
        ["lower_conf", -2.440432203804888],
    ]
)
def test_evaluate_model(type, result):
    """
    test that method '_evaluate_model' works
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    # define training data
    train_X = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=dtype, device=device)
    train_Y = torch.tensor([[1], [1.5]], dtype=dtype, device=device)

    # define and train model
    mymodel = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    ll = ExactMarginalLogLikelihood(mymodel.likelihood, mymodel)
    fit_gpytorch_model(ll)

    # construct temporary class
    class Tmp:
        def __init__(self, mymodel):
            self.model = {"model": mymodel}

        from greattunes._best_response import _evaluate_model

    cls = Tmp(mymodel)

    # data point to investigate
    x = torch.tensor([[2, 3, 4]], dtype=dtype, device=device)

    # run method
    y = cls._evaluate_model(type=type, x=x)

    # assert
    assert isinstance(y, torch.DoubleTensor)

    ERROR_LIM = 1e-8
    assert abs(y.item()-result) <= ERROR_LIM
