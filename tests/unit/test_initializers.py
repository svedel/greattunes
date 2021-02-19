"""
This file contains a helper class with initializer methods. It is only wrapped as a class for convenience. Will
therefore only test individual functions
"""
import math
import pytest
import torch
import warnings
from creative_project._initializers import Initializers
import creative_project._best_response
from creative_project.utils import DataSamplers


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


def test_Initializers__initialize_best_response_first_add(custom_models_simple_training_data_4elements,
                                                          tmp_Initializers_with_find_max_response_value_class,
                                                          monkeypatch):
    """
    test initialization of best response data structures based on input data
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # create test version of Initializers to endow it with the property from _find_max_response_value, which is
    # otherwise defined as a static method in ._best_response
    cls = tmp_Initializers_with_find_max_response_value_class

    # monkeypatch function call
    tmp_output = 1.0
    def mock_find_max_response_value(train_X, train_Y):
        return torch.tensor([[tmp_output]], dtype=torch.double), torch.tensor([[tmp_output]], dtype=torch.double)
    monkeypatch.syspath_prepend("..")
    monkeypatch.setattr(
        #creative_project._best_response, "_find_max_response_value", mock_find_max_response_value
        cls, "_find_max_response_value", mock_find_max_response_value
    )

    # initialize class and register required attributes
    #cls = Initializers()
    cls.train_X = train_X
    cls.train_Y = train_Y

    # define required attributes for test to pass (IRL set in CreativeProject which is a child of Initializers)
    cls.dtype = torch.double
    cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def test_Initializers__initialize_training_data_unit(custom_models_simple_training_data_4elements, monkeypatch):
    """
    test '__initialize_training_data_unit' with all modes that fail together with the one that passes. Monkeypatch
    dependency to '_validators.Validators.__validate_training_data'
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    ### First test: iterations have already started (no initialization accepted)
    # initialize
    cls = Initializers()
    cls.model = {
        "covars_sampled_iter": 1,
        "response_sampled_iter": 1
    }

    # run the method
    cls._Initializers__initialize_training_data(train_X=train_X, train_Y=train_Y)

    # assert that nothing has run
    assert cls.train_X == None
    assert cls.train_Y == None
    assert cls.proposed_X == None

    ### Second test: iterations not started, None data provided (no initialization accepted)
    # initialize
    cls = Initializers()
    cls.model = {
        "covars_sampled_iter": 0,
        "response_sampled_iter": 0
    }

    # run the method
    cls._Initializers__initialize_training_data(train_X=None, train_Y=None)

    # assert that nothing has run
    assert cls.train_X == None
    assert cls.train_Y == None
    assert cls.proposed_X == None

    ### Third test: iterations not started, data provided, monkeypatching __validate_training_data
    # (initialization accepted)
    # initialize
    cls = Initializers()
    cls.model = {
        "covars_sampled_iter": 0,
        "response_sampled_iter": 0
    }

    # monkeypatching
    def mock__validate_training_data(train_X, train_Y):
        return True
    monkeypatch.setattr(
        cls, "_Validators__validate_training_data", mock__validate_training_data
    )

    # run the method
    cls._Initializers__initialize_training_data(train_X=train_X, train_Y=train_Y)

    # assert that the data has been validated and stored in right places
    for it in range(train_X.shape[0]):
        assert cls.train_X[it].item() == train_X[it].item()
        assert cls.train_Y[it].item() == train_Y[it].item()
        assert cls.proposed_X[it].item() is not None  # proposed_X is being set to torch.empty (a random number)


@pytest.mark.parametrize("covars",
                         [
                             [(1, 0, 2)],
                             [(0.5, 0, 1), (12.5, 8, 15), (-2, -4, 1.1)]
                         ]
                         )
def test_Initializers_determine_number_random_samples(covars):
    """
    test that 'determine_number_random_samples' returns the right number (int(round(sqrt(d))) where d is number of
    covariates. First test should return 1, second return 2
    :return:
    """

    # initialize class
    cls = Initializers()

    # set attribute
    cls.covars = covars

    # get result
    num_random = cls.determine_number_random_samples()

    # assert
    assert num_random == int(round(math.sqrt(len(covars)), 0))


@pytest.mark.parametrize(
    "train_X, train_Y, random_start, num_initial_random, random_sampling_method, num_initial_random_points_res, sampling_method_res",
    [
        [torch.tensor([[1, 2, 3]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), False, None, None, 0, None], # Case 1
        [torch.tensor([[1, 2, 3]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), False, 2, "random", 0, None], # Case 1. Special twist to ensure correct behavior even if user sets random parameter features while still choosing against random start
        [torch.tensor([[1, 2, 3]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), True, None, None, 33, "latin_hcs"], # Case 2. 33 is the output from the method 'determine_number_random_samples' which is being monkeypatched (see below)
        [torch.tensor([[1, 2, 3]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), True, 2, "random", 2, "random"], # Case 2. Special case where both number of samples and sampling method are set by user
        [None, None, False, None, None, 33, "latin_hcs"], # Case 3
        [None, None, False, 2, "random", 33, "random"], # Case 3
        [None, None, True, None, None, 33, "latin_hcs"], # Case 4
        [None, None, True, 12, "random", 12, "random"], # Case 4
    ]
)
def test_Initializers__initialize_random_start_works(train_X, train_Y, random_start, num_initial_random,
                                                     random_sampling_method, num_initial_random_points_res,
                                                     sampling_method_res, monkeypatch):
    """
    test that initialization of random start works. Monkeypatching 'determine_number_random_samples' which determines
    number of samples if nothing provided by user.
    There are 4 cases to consider (case -> expected behavior)
    - CASE 1: train_X, train_Y present; random_start = False -> (self.num_initial_random_points = 0, self.random_sampling_method = None)
    - CASE 2: train_X, train_Y present; random_start = True -> (self.num_initial_random_points = user-provided number / round(sqrt(num_covariates)), self.random_sampling_method = user-provided)
    - CASE 3: train_X, train_Y NOT present; random_start = False -> (this is a case of INCONSISTENCY in user input. Expect user has made a mistake so proceeds but throws warning. self.num_initial_random_points = round(sqrt(num_covariates)), self.random_sampling_method = user-provided)
    - CASE 4: train_X, train_Y NOT present; random_start = True -> (self.num_initial_random_points = user-provided number / round(sqrt(num_covariates)), self.random_sampling_method = user-provided)
    """

    # initialize class
    cls = Initializers()

    # start by monkeypatching
    NUM_RETURN = 33
    def mock_determine_number_random_samples():
        return NUM_RETURN
    monkeypatch.setattr(cls, "determine_number_random_samples", mock_determine_number_random_samples)

    # set attributes
    cls.train_Y = train_Y
    cls.train_X = train_X

    # run method
    cls._Initializers__initialize_random_start(random_start=random_start, num_initial_random=num_initial_random,
                                  random_sampling_method=random_sampling_method)

    # assert outcome
    assert cls.num_initial_random_points == num_initial_random_points_res
    assert cls.random_sampling_method == sampling_method_res


def test_Initializers__initialize_random_start_fails(monkeypatch):
    """
    test that '__initialize_random_start' if nonexistent sampling method provided via input 'random_sampling_method'
    """

    # initialize class
    cls = Initializers()

    # start by monkeypatching
    NUM_RETURN = 33

    def mock_determine_number_random_samples():
        return NUM_RETURN

    monkeypatch.setattr(cls, "determine_number_random_samples", mock_determine_number_random_samples)

    # set attributes
    cls.train_Y = None
    cls.train_X = None

    # for reference, the list of acceptable sampling methods
    SAMPLING_METHODS_LIST = [func for func in dir(DataSamplers) if
                             callable(getattr(DataSamplers, func)) and not func.startswith("__")]

    # run method
    with pytest.raises(Exception) as e:
        cls._Initializers__initialize_random_start(random_start=True, num_initial_random=2,
                                                   random_sampling_method="junk")  # acceptable values of 'random_sampling_method' given in SAMPLING_METHODS_LIST
    assert str(e.value) == "creative_project._initializers.Initializers.__initialize_random_start: The parameter 'random_sampling_method' is not among allowed values ('" + "', '".join(SAMPLING_METHODS_LIST) + "')."


@pytest.mark.parametrize(
    "train_X, train_Y, random_start, num_initial_random, random_sampling_method, num_initial_random_points_res, sampling_method_res",
    [
        [torch.tensor([[1, 2, 3]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), False, 2, "random", 0, None], # Case 1. Special twist to ensure correct behavior even if user sets random parameter features while still choosing against random start
    ]
)
def test_Initializers__initialize_random_start_warning(train_X, train_Y, random_start, num_initial_random,
                                                     random_sampling_method, num_initial_random_points_res,
                                                     sampling_method_res, monkeypatch):
    """
    test that the right warning message is raised
    """

    # initialize class
    cls = Initializers()

    # start by monkeypatching
    NUM_RETURN = 33

    def mock_determine_number_random_samples():
        return NUM_RETURN

    monkeypatch.setattr(cls, "determine_number_random_samples", mock_determine_number_random_samples)

    # set attributes
    cls.train_Y = train_Y
    cls.train_X = train_X

    # run method
    cls._Initializers__initialize_random_start(random_start=random_start, num_initial_random=num_initial_random,
                                           random_sampling_method=random_sampling_method)

    # test that warning is raised
    my_warning = "Inconsistent settings for optimization initialization: No initial data provided via 'train_X' and 'train_Y' but also 'random_start' is set to 'False'. Adjusting to start with " + str(num_initial_random) + " random datapoints."
    with pytest.warns(UserWarning):
        warnings.warn(my_warning, UserWarning)
