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


@pytest.mark.parametrize(
    "x_tuple, tuple_datatype_result",
    [
        [(1, 0, 3), int],  # test int
        [(2.2, 1.1, 3.3), float],  # test float
        [(2, 1, 3.3), float],  # test float in case where only 1 float (should recast ints as float)
        [("str", "hej"), str],  # test str (not 3 elemets)
        [("hej", "med", "dig", "din", "fisk"), str],  # test str (not 3 elements)
        [(1, 0.2, "hej"), str],  # test str (should default and cast everything as str)
    ]
)
def test__determine_tuple_datatype_unit_works(x_tuple, tuple_datatype_result):
    """
    tests that method __determine_tuple_datatype works in terms of assigning the right data type for tuple content.
    test for both a) cases where all entries are the same type (str and other), b) cases where either a float or a str
    is present (in which case it should default to float or str)
    """

    # initialize class
    cls = Initializers()

    # run the method and assert that the right type is returned
    tuple_datatype = cls._Initializers__determine_tuple_datatype(x_tuple)

    assert tuple_datatype == tuple_datatype_result


@pytest.mark.parametrize(
    "x_tuple, error_msg",
    [
        [(1, 0, True), "creative_project._initializers.Initialzer.__determine_tuple_datatype: individual covariates provided via tuples can only be of types ('float', 'int', 'str') but was provided " + str(type(True))],
        [({}, 1.1, 3.3), "creative_project._initializers.Initialzer.__determine_tuple_datatype: individual covariates provided via tuples can only be of types ('float', 'int', 'str') but was provided " + str(type({}))],  # test float
        [("str", 1, []), "creative_project._initializers.Initialzer.__determine_tuple_datatype: individual covariates provided via tuples can only be of types ('float', 'int', 'str') but was provided " + str(type([]))],  # test float in case where only 1 float (should recast ints as float)
    ]
)
def test__determine_tuple_datatype_unit_fails(x_tuple, error_msg):
    """
    tests that method __determine_tuple_datatype throws the correct error in case a tuple-element of data type other
    than int, float or str is provided
    """

    # initialize class
    cls = Initializers()

    # run the method and assert that the right type is returned
    with pytest.raises(Exception) as e:
        tuple_datatype = cls._Initializers__determine_tuple_datatype(x_tuple)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "covars, total_num_covars, covar_mapped_names, GP_kernel_mapping_covar_identification, covar_details",
    [
        [[(1, 0, 2)], 1, ["covar0"], [{"type": int, "column": [0]}], {"covar0":{"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}}],
        [[(1, 0, 2), ("hej", "med", "dig", "hr")], 5, ["covar0", "covar1_hej", "covar1_med", "covar1_dig", "covar1_hr"], [{"type": int, "column": [0]}, {"type": str, "column": [1, 2, 3, 4]}], {"covar0":{"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}, "covar1": {"guess": "hej", "options": {"hej", "med", "dig", "hr"}, "type": str, "columns": [1, 2, 3, 4], "opt_names": ["covar1_hej", "covar1_med", "covar1_dig", "covar1_hr"]}}],
    ]
)
def test__initialize_covars_list_of_tuples_works(covars, total_num_covars, covar_mapped_names,
                                                 GP_kernel_mapping_covar_identification, covar_details, monkeypatch):
    """
    test that the right attributes 'covar_details', 'GP_kernel_mapping_covar_identification', 'covar_mapped_names' and
    'total_num_covars' are created correctly. Monkeypatches '__determine_tuple_datatype', '_Validators__validate_covars'
    and '_Validators__validate_num_entries_covar_tuples'. Tests for both single and multiple-covariate case, and also
    tests both numerical and categorical covariates

    the method "__initialize_covars_list_of_tuples" takes list of tuples as input
    """

    # initialize class
    cls = Initializers()

    # monkeypatches
    def mock__determine_tuple_datatype(tpl):  # beware that we use data type of first element tuple for whole tuple
        return type(tpl[0])
    monkeypatch.setattr(cls, "_Initializers__determine_tuple_datatype", mock__determine_tuple_datatype)

    def mock__validate_covars(covars):
        return True
    monkeypatch.setattr(cls, "_Validators__validate_covars", mock__validate_covars)

    def mock__validate_num_entries_covar_tuples(covars, covars_tuple_datatypes):
        return True
    monkeypatch.setattr(cls, "_Validators__validate_num_entries_covar_tuples", mock__validate_num_entries_covar_tuples)

    # run the method
    cls._Initializers__initialize_covars_list_of_tuples(covars=covars)

    # assert
    assert cls.total_num_covars == total_num_covars
    assert cls.GP_kernel_mapping_covar_identification == GP_kernel_mapping_covar_identification
    assert cls.covar_mapped_names == covar_mapped_names
    assert cls.covar_details == covar_details


@pytest.mark.parametrize(
    "covars, total_num_covars, covar_mapped_names, GP_kernel_mapping_covar_identification, covar_details",
    [
        [{"var0": {"guess":1, "min": 0, "max": 2, "type": int}}, 1, ["var0"], [{"type": int, "column": [0]}], {"var0":{"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}}],
        [{"var0": {"guess":1, "min": 0, "max": 2, "type": int}, "idiot": {"guess": "hej", "options": {"hej", "med", "dig", "hr"}, "type": str}}, 5, ["var0", "idiot_hej", "idiot_med", "idiot_dig", "idiot_hr"], [{"type": int, "column": [0]}, {"type": str, "column": [1, 2, 3, 4]}], {"var0": {"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}, "idiot": {"guess": "hej", "options": {"hej", "med", "dig", "hr"}, "type": str, "columns": [1, 2, 3, 4], "opt_names": ["idiot_hej", "idiot_med", "idiot_dig", "idiot_hr"]}}],
    ]
)
def test__initialize_covars_dict_of_dicts_works(covars, total_num_covars, covar_mapped_names,
                                                GP_kernel_mapping_covar_identification, covar_details):
    """
    test that the right attributes 'covar_details', 'GP_kernel_mapping_covar_identification', 'covar_mapped_names' and
    'total_num_covars' are created correctly. Tests for both single and multiple-covariate case, and also
    tests both numerical and categorical covariates

    the method "__initialize_covars_dict_of_dicts" takes a dict of dicts as input
    """

    # initialize class
    cls = Initializers()

    # run method
    cls._Initializers__initialize_covars_dict_of_dicts(covars=covars)

    # assert
    assert cls.total_num_covars == total_num_covars
    assert cls.GP_kernel_mapping_covar_identification == GP_kernel_mapping_covar_identification
    assert set(cls.covar_mapped_names) == set(covar_mapped_names)
    # investigate the different layers of covar_details
    assert cls.covar_details.keys() == covar_details.keys()
    for k in cls.covar_details.keys():
        for kk in cls.covar_details[k].keys():
            # special attention to list with key "opt_names" because it is generated from a set in the method meaning
            # that the order of the list entries can shuffle
            if kk == "opt_names":
                assert set(cls.covar_details[k][kk]) == set(covar_details[k][kk])
            else:
                assert cls.covar_details[k][kk] == covar_details[k][kk]


@pytest.mark.parametrize(
    "covars, error_msg",
    [
        [{'var0': {'guess': 1, 'min': 0, 'max': 2}, 'var1': (1, 0, 3)}, "creative_project._initializers.Initializers.__initialize_covars_dict_of_dicts: 'covars' provided as part of class initialization must be either a list of tuples or a dict of dicts. Current provided is a dict containing data types {<class 'dict'>, <class 'tuple'>}."],  # incorrect data type in 'covars'
        [{'var0': {'guess': 1, 'min': 0, 'max': 2}}, "creative_project._initializers.Initializers.__initialize_covars_dict_of_dicts: key 'type' missing for covariate 'var0' (covars['var0']={'guess': 1, 'min': 0, 'max': 2})."], # should fail for not having element "type"
        [{'var0': {'guess': 1, 'max': 2, 'type': int}}, "creative_project._initializers.Initializers.__initialize_covars_dict_of_dicts: key 'min' missing for covariate 'var0' (covars['var0']={'guess': 1, 'max': 2, 'type': <class 'int'>})."], # should fail for missing 'min'
        [{'var0': {'guess': 1, 'min': 0, 'type': int}}, "creative_project._initializers.Initializers.__initialize_covars_dict_of_dicts: key 'max' missing for covariate 'var0' (covars['var0']={'guess': 1, 'min': 0, 'type': <class 'int'>})."], # should fail for missing 'max'
        [{'var0': {'guess': 1, 'max': 2, 'type': int}, 'var1': {'guess': 'red', 'options':{'red', 'green'}, 'type': str}}, "creative_project._initializers.Initializers.__initialize_covars_dict_of_dicts: key 'min' missing for covariate 'var0' (covars['var0']={'guess': 1, 'max': 2, 'type': <class 'int'>})."], # should fail for missing 'min' even though more variables provided
        [{'var0': {'options': {'red', 'blue'}, 'type': str}}, "creative_project._initializers.Initializers.__initialize_covars_dict_of_dicts: key 'guess' missing for covariate 'var0' (covars['var0']={'options': {'red', 'blue'}, 'type': <class 'str'>})."],  # missing 'options"
    ]
)
def test__initialize_covars_dict_of_dicts_fails(covars, error_msg):
    """
    test that the right error message is returned if inconsistent data is provided to dict of dicts 'covars' for method
    '__initialize_covars_dict_of_dicts'. Tests for both single and multiple-covariate case, and also tests both
    numerical and categorical covariates

    the method "__initialize_covars_dict_of_dicts" takes a dict of dicts as input
    """

    # initialize class
    cls = Initializers()

    # run method and assert
    with pytest.raises(Exception) as e:
        cls._Initializers__initialize_covars_dict_of_dicts(covars=covars)
    assert str(e.value) == error_msg