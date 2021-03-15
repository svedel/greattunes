import pandas as pd
import pytest
import torch
from creative_project import CreativeProject


@pytest.mark.parametrize(
    "covars, error_msg",
    [
        [None, "creative_project._initializers.Initializers.__initialize_from_covars: provided 'covars' is of type <class 'NoneType'> but must be of types {'list', 'dict'}."],
        # checks no covars data provided
        [(1, 2, 3),
         "creative_project._initializers.Initializers.__initialize_from_covars: provided 'covars' is of type <class 'tuple'> but must be of types {'list', 'dict'}."],
        # checks fail if covars not a list of tuples
        [[1, 2, 3],
         "kre8_core.creative_project._validators.Validator.__validate_covars: entry in covars list is not tuple"],
        # checks that covars is a list of tuples
    ])
def test_CreativeProject__init__covars_notrainingdata_fails_functional(covars, error_msg):
    """
    test CreativeProject class initialization under conditions where it should work (no data provided)
    """

    with pytest.raises(Exception) as e:
        # set up class
        assert CreativeProject(covars=covars)
    assert str(e.value) == error_msg


def test_CreativeProject__init__covars_trainingdata_works_functional(
        covars_for_custom_models_simple_training_data_4elements,
        custom_models_simple_training_data_4elements):
    """
    tests class initialization with training data provided
    """

    covars = covars_for_custom_models_simple_training_data_4elements
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # initialize class
    cls = CreativeProject(covars=covars, train_X=train_X, train_Y=train_Y)

    # assert covars are set up correctly
    assert cls.initial_guess[0].item() == covars[0][0]
    assert cls.covar_bounds[0][0].item() == covars[0][1]

    # assert training data set
    assert cls.train_X is not None
    assert cls.train_X[0].item() == train_X[0].item()
    assert cls.train_Y is not None
    assert cls.train_Y[1].item() == train_Y[1].item()

    # assert best response initialized; look at two different spots
    assert cls.covars_best_response_value[1].item() == -1.0
    assert cls.best_response_value[1].item() == 0.2
    assert cls.covars_best_response_value[3].item() == 1.0
    assert cls.best_response_value[3].item() == 2.0


def test_CreativeProject__init__covars_trainingdata_multivariate_works_functional(training_data_covar_complex):
    """
    passing test with multivariate initialization, multiple observations. test that class initialization works with
    multivariate train_X data (3 covars) with multiple observations (3)

    also investigate that pretty data x_data and y_data is created
    """

    # data
    covars = training_data_covar_complex[0]
    train_X = training_data_covar_complex[1]
    train_Y = training_data_covar_complex[2]

    # initialize class
    cls = CreativeProject(covars=covars, train_X=train_X, train_Y=train_Y)

    # assert covars are set up correctly
    assert cls.initial_guess[0][1].item() == covars[1][0]
    assert cls.covar_bounds[0][0].item() == covars[0][1]

    # assert training data set
    assert cls.train_X is not None
    assert cls.train_X[0, 2].item() == train_X[0, 2].item()
    assert cls.train_X[2, 1].item() == train_X[2, 1].item()
    assert cls.train_Y is not None
    assert cls.train_Y[1].item() == train_Y[1].item()

    print("cls.covars_best_response_value")
    print(cls.covars_best_response_value)

    # assert best response initialized; look at two different spots (rows)
    # with multivariate train_X, "covars_best_response_value" is a row at each iteration.
    # Best response is at index 1 (second element in train_Y), so best train_X and best train_Y at last element (index
    # 2) is the same as train_X, train_Y at index 1
    for it in range(train_X.shape[1]):
        assert cls.covars_best_response_value[0, it].item() == train_X[0, it].item()
    assert cls.best_response_value[0].item() == train_Y[0].item()

    for it in range(train_X.shape[1]):
        assert cls.covars_best_response_value[2, it].item() == train_X[1, it].item()
    assert cls.best_response_value[2].item() == train_Y[1].item()

    # assert that pretty data x_data, y_data is created
    assert hasattr(cls, "x_data")
    assert hasattr(cls, "y_data")

    tmp_x_data = cls.x_data.values

    # assert that the numbers are correct in pretty data
    for i in range(train_X.shape[0]):
        assert cls.y_data["Response"].iloc[i] == train_Y[i,0].item()
        for j in range(train_X.shape[1]):
            assert tmp_x_data[i,j] == train_X[i,j].item()


def test_CreativeProject__init__covars_trainingdata_multivariate_fails_functional(training_data_covar_complex):
    """
    failing test with multivariate initialization, multiple observations. test that class initialization fails with
    multivariate train_X data (4 covars, expect 3) with multiple observations (3)
    """

    # data
    covars = training_data_covar_complex[0]
    train_X_tmp = training_data_covar_complex[1]

    train_Y = training_data_covar_complex[2]
    train_X = torch.cat((train_X_tmp, torch.tensor([[-1.1], [-0.5], [-2.2]], dtype=torch.double)), dim=1)

    cls = CreativeProject(covars=covars, train_X=train_X, train_Y=train_Y)

    assert cls.train_X is None
    assert cls.train_Y is None


def test_CreativeProject__init__covars_dict_of_dicts_works():
    """
    test that initializing the whole framework from a dict of dict works. also test that pretty data for train_X, train_Y
    with categorical variable works
    """

    covars = [(1, 0, 3), (1.1, -1.2, 3.4), ("red", "green", "blue")]
    train_X = torch.tensor([[2, -0.7, 1.0, 0.0, 0.0], [1, 1.1, 0.0, 0.0, 1.0]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    train_X_interrogate = pd.DataFrame({'covar0':[2, 1], 'covar1': [-0.7, 1.1], 'covar2': ["red", "blue"]})
    train_Y = torch.tensor([[1.1], [3.5]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    cls = CreativeProject(covars=covars, train_X=train_X, train_Y=train_Y)

    # test that attributes have been created
    assert hasattr(cls, "train_X")
    assert hasattr(cls, "train_Y")
    assert hasattr(cls, "x_data")
    assert hasattr(cls, "y_data")
    assert hasattr(cls, "covar_details")

    # assert x_data, y_data
    tmp_xdata = cls.x_data.values
    tmp_x_interrogate = train_X_interrogate.values
    for i in range(2):
        assert cls.y_data["Response"].iloc[i] == train_Y[i,0].item()
        for j in range(3):
            assert tmp_xdata[i,j] == tmp_x_interrogate[i,j]



@pytest.mark.parametrize(
    "random_start, num_initial_random, random_sampling_method, train_X, train_Y, num_samples_res, method_sampling_res",
    [
        [True, None, "latin_hcs", None, None, 2, "latin_hcs"],  # because we have 3 covariates (see below), and no number of initial random samples have been provided, the method will default to round(sqrt(3)) = 2
        [True, 3, "random", torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.double), torch.tensor([[43], [44]], dtype=torch.double), 3, "random"],  # here expect the number of initial points to be inherited from input
    ]
)
def test_CreativeProject__init__initialize_random_start_works(random_start, num_initial_random, random_sampling_method,
                                                              train_X, train_Y, num_samples_res, method_sampling_res):
    """
    test that initialization of random start parameters works. There are 4 different cases treated by
    '__initialize_random_start', which are exhaustively tested by unit and integration tests for _initializers.py. Hence
    purpose of present tests is only to ensure that the method also works when initalizing from CreativeProject, and
    thus we are running with reduced test load
    """

    # data
    covars = [(1, 0, 4), (3.4, -1.2, 6), (12, 11, 17.8)]

    # initialize class
    cls = CreativeProject(covars=covars, train_X=train_X, train_Y=train_Y, random_start=random_start,
                          num_initial_random=num_initial_random, random_sampling_method=random_sampling_method)

    # assert
    assert cls.random_sampling_method == method_sampling_res
    assert cls.num_initial_random_points == num_samples_res
