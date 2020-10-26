import pytest
from creative_project import CreativeProject


@pytest.mark.parametrize(
    "covars, error_msg",
    [
        [None, "kre8_core.creative_project._validators.Validator.__validate_covars: covars is None"],  # checks no covars data provided
        [(1, 2, 3), "kre8_core.creative_project._validators.Validator.__validate_covars: covars is not list of tuples (not list)"],  # checks fail if covars not a list of tuples
        [[1, 2, 3], "kre8_core.creative_project._validators.Validator.__validate_covars: entry in covars list is not tuple"],  # checks that covars is a list of tuples
        [[("hej", 2, 3)], "kre8_core.creative_project._validators.Validator.__validate_covars: tuple element hej in covars list is neither of type float or int"]  # test that all elements in tuples are of type int or float
    ])
def test_CreativeProject__init__covars_notrainingdata_fails_functional(covars, error_msg):
    """
    test CreativeProject class initialization under conditions where it should work (no data provided)
    """

    with pytest.raises(Exception) as e:
        # set up class
        assert CreativeProject(covars=covars)
    assert str(e.value) == error_msg


def test_CreativeProject__init__covars_trainingdata_works_functional(covars_for_custom_models_simple_training_data_4elements,
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


# passing test with multivariate initialization, multiple observations
def test_CreativeProject__init__covars_trainingdata_multivariate_works_functional(training_data_covar_complex):
    """
    test that class initialization works with multivariate train_X data (3 covars) with multiple observations (3)
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
