from creative_project._initializers import Initializers
from creative_project._best_response import _find_max_response_value
import pandas as pd
import pytest
import torch


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

    # assert some of the new attributes created by __initialize_from_covars
    assert hasattr(cls, "covar_details")
    assert hasattr(cls, "GP_kernel_mapping_covar_identification")
    assert hasattr(cls, "covar_mapped_names")
    assert hasattr(cls, "total_num_covars")

    # check that 1 covariate provided for dataset_id = 0, 3 for dataset_id = 1
    if dataset_id == 0:
        assert cls.total_num_covars == 1
        assert cls.covar_mapped_names == ["covar0"]
        assert "covar0" == list(cls.covar_details.keys())[0]
        assert cls.covar_details["covar0"]["type"] == float
    elif dataset_id == 1:
        cvlist = ["covar0", "covar1", "covar2"]

        assert cls.total_num_covars == 3
        assert cls.covar_mapped_names == cvlist

        for i in cls.covar_details.keys():
            assert i in cvlist
            assert cls.covar_details[i]["type"] == float


@pytest.mark.parametrize(
    "covars, total_num_covars, covar_mapped_names, GP_kernel_mapping_covar_identification, covar_details",
    [
        [{"var0": {"guess":1, "min": 0, "max": 2, "type": int}}, 1, ["var0"], [{"type": int, "column": [0]}], {"var0":{"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}}],
        [{"var0": {"guess":1, "min": 0, "max": 2, "type": int}, "idiot": {"guess": "hej", "options": {"hej", "med", "dig", "hr"}, "type": str}}, 5, ["var0", "idiot_hej", "idiot_med", "idiot_dig", "idiot_hr"], [{"type": int, "column": [0]}, {"type": str, "column": [1, 2, 3, 4]}], {"var0": {"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}, "idiot": {"guess": "hej", "options": {"hej", "med", "dig", "hr"}, "type": str, "columns": [1, 2, 3, 4], "opt_names": ["idiot_hej", "idiot_med", "idiot_dig", "idiot_hr"]}}],
    ]
)
def test__initialize_from_covars_dict_of_dicts_works(covars, total_num_covars, covar_mapped_names,
                                                GP_kernel_mapping_covar_identification, covar_details):
    """
    asserts that __initialize_from_covars works when provided dict of dicts as input

    test that the right attributes 'covar_details', 'GP_kernel_mapping_covar_identification', 'covar_mapped_names' and
    'total_num_covars' are created correctly. Tests for both single and multiple-covariate case, and also
    tests both numerical and categorical covariates. Monkeypatches validation (taken from Validators parent class, unit
    tested there)
    """

    # number of covariates (= number of columns)
    num_cols = total_num_covars

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

    # assert some of the new attributes created by __initialize_from_covars
    assert hasattr(cls, "covar_details")
    assert hasattr(cls, "GP_kernel_mapping_covar_identification")
    assert hasattr(cls, "covar_mapped_names")
    assert hasattr(cls, "total_num_covars")

    # assert details of created attributes
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


def test_Initializers__initialize_best_response_functional(custom_models_simple_training_data_4elements,
                                                           custom_models_simple_training_data_4elements_covar_details,
                                                           tmp_Initializers_with_find_max_response_value_class):
    """
    test initialization of best response data structures based on input data
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # covar details
    covar_details = custom_models_simple_training_data_4elements_covar_details[0]
    covar_mapped_names = custom_models_simple_training_data_4elements_covar_details[1]

    # create test version of Initializers to endow it with the property from _find_max_response_value, which is
    # otherwise defined as a static method in ._best_response
    #class TmpClass(Initializers):
    #    from creative_project._best_response import _find_max_response_value
    #cls = TmpClass()
    cls = tmp_Initializers_with_find_max_response_value_class

    # initialize class and register required attributes
    #cls = Initializers()
    #
    # add validation method, only needed for test purposes
    #cls._find_max_response_value = _find_max_response_value

    cls.train_X = train_X
    cls.train_Y = train_Y

    # add attributes
    cls.covar_details = covar_details
    cls.covar_mapped_names = covar_mapped_names

    # define required attributes for test to pass (IRL set in CreativeProject which is a child of Initializers)
    cls.dtype = torch.double
    cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run the function
    cls._Initializers__initialize_best_response()

    # assert that the function has been applied
    assert isinstance(cls.covars_best_response_value, torch.Tensor)
    assert isinstance(cls.best_response_value, torch.Tensor)

    # assert that pretty format data structures have been created
    assert isinstance(cls.covars_best_response, pd.DataFrame)
    assert isinstance(cls.best_response, pd.DataFrame)

    # check size
    assert cls.covars_best_response_value.shape[0] == train_X.shape[0]
    assert cls.best_response_value.shape[0] == train_Y.shape[0]

    # check values, compare to mock_find_max_response_value
    # below two "test"-tensors contain best values corresponding to data from
    # conftest.py's "custom_models_simple_training_data_4elements"
    test_max_covars_test = torch.tensor([[-1.0], [-1.0], [-0.5], [1.0]], dtype=torch.double)
    test_max_response_test = torch.tensor([[0.2], [0.2], [0.5], [2.0]], dtype=torch.double)
    for it in range(train_X.shape[0]):

        # test that tensor-format best response and covars has right values
        assert cls.covars_best_response_value[it].item() == test_max_covars_test[it].item()
        assert cls.best_response_value[it].item() == test_max_response_test[it].item()

        # test that pretty-format best response and covars has right values
        assert cls.covars_best_response["covar0"].iloc[it] == test_max_covars_test[it].item()
        assert cls.best_response["Response"].iloc[it] == test_max_response_test[it].item()


def test_Initializers__initialize_training_data_functional(custom_models_simple_training_data_4elements,
                                                           covars_for_custom_models_simple_training_data_4elements):
    """
    test private method initialize_training_data which depends on private method from parent class Validators. Test
    only functional and non-functional via provided initial data train_X, train_Y
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]
    covars = covars_for_custom_models_simple_training_data_4elements

    ### First test: it passes (use train_X, train_Y)
    # initialize class and register required attributes
    cls = Initializers()
    cls.model = {  # set required attributes
        "covars_sampled_iter": 0,
        "response_sampled_iter": 0
    }
    cls.initial_guess = torch.tensor([[g[0] for g in covars]], dtype=torch.double)

    # run the method
    cls._Initializers__initialize_training_data(train_X=train_X, train_Y=train_Y)

    # assert that the data has been validated and stored in right places
    for it in range(train_X.shape[0]):
        assert cls.train_X[it].item() == train_X[it].item()
        assert cls.train_Y[it].item() == train_Y[it].item()
        assert cls.proposed_X[it].item() is not None  # proposed_X is being set to torch.empty (a random number)

    ### First test: it passes (use train_X, train_Y)
    # initialize class and register required attributes
    cls = Initializers()
    cls.model = {  # set required attributes
        "covars_sampled_iter": 0,
        "response_sampled_iter": 0
    }

    # run the method
    cls._Initializers__initialize_training_data(train_X=None, train_Y=None)

    # assert that nothing has run
    assert cls.train_X == None
    assert cls.train_Y == None
    assert cls.proposed_X == None


@pytest.mark.parametrize(
    "train_X, train_Y, initial_guess, train_X_created, train_Y_created",
    [
        [pd.DataFrame({"a": [1], "b":[0.2], "c":["red"]}), pd.DataFrame({"Response": [1.2]}), [0, 2.2, 1.0, 0.0, 0.0], torch.tensor([[1, 0.2, 1.0, 0.0, 0.0]], dtype=torch.double), torch.tensor([[1.2]], dtype=torch.double)]
    ]
)
def test_initialize_training_data_and_pretty_data(train_X, train_Y, initial_guess, train_X_created,
                                                              train_Y_created, covar_details_covar_mapped_names):
    """
    test that initialization of training data works when provided in pretty format

    test for all data types (integer, continuous, categorical)
    """

    # initialize class and register required attributes
    cls = Initializers()
    cls.model = {  # set required attributes
        "covars_sampled_iter": 0,
        "response_sampled_iter": 0
    }
    cls.initial_guess = torch.tensor([initial_guess], dtype=torch.double)
    cls.covar_details = covar_details_covar_mapped_names[0]
    cls.covar_mapped_names = covar_details_covar_mapped_names[1]
    cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run the method to initialize training data
    cls._Initializers__initialize_training_data(train_X=train_X, train_Y=train_Y)

    # run the method to initialize pretty data
    cls.x_data, cls.y_data = cls._Initializers__initialize_pretty_data()

    # test train_X, train_Y created
    for i in range(cls.train_X.size()[0]):
        assert cls.train_Y[i,0].item() == train_Y_created[i,0].item()
        for j in range(cls.train_X.size()[1]):
            assert cls.train_X[i, j].item() == train_X_created[i, j].item()

    # test that x_data and y_data are equal to initially provided train_X, train_Y (these provided in pandas)
    xdf_tmp = train_X.values
    ydf_tmp = train_Y.values

    xret_tmp = cls.x_data.values
    yret_tmp = cls.y_data.values

    for i in range(xret_tmp.shape[0]):
        assert ydf_tmp[i,0] == yret_tmp[i,0]
        for j in range(xret_tmp.shape[1]):
            assert xdf_tmp[i,j] == xret_tmp[i,j]


@pytest.mark.parametrize(
    "train_X, train_Y, covars, random_start, num_initial_random, random_sampling_method, num_initial_random_points_res, sampling_method_res",
    [
        [torch.tensor([[1, 2, 3]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), [(1,0,2), (2, 1, 3), (3, 2, 4)], False, None, None, 0, None], # Case 1
        [torch.tensor([[1, 2, 3]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), [(1,0,2), (2, 1, 3), (3, 2, 4)], False, 2, "random", 0, None], # Case 1. Special twist to ensure correct behavior even if user sets random parameter features while still choosing against random start
        [torch.tensor([[1, 2, 3]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), [(1,0,2), (2, 1, 3), (3, 2, 4)], True, None, None, 2, "latin_hcs"], # Case 2. 2 is the output from the method 'determine_number_random_samples' since its round(sqrt(3)), where 3 is the number of covariates in train_X and covars
        [torch.tensor([[1, 2, 3]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), torch.tensor([[22]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")), [(1,0,2), (2, 1, 3), (3, 2, 4)], True, 12, "random", 12, "random"], # Case 2. Special case where both number of samples and sampling method are set by user
        [None, None, [(1,0,2), (2, 1, 3), (3, 2, 4)], False, None, None, 2, "latin_hcs"], # Case 3
        [None, None, [(1,0,2), (2, 1, 3), (3, 2, 4)], False, 12, "random", 2, "random"], # Case 3
        [None, None, [(1,0,2), (2, 1, 3), (3, 2, 4)], True, None, None, 2, "latin_hcs"], # Case 4
        [None, None, [(1,0,2), (2, 1, 3), (3, 2, 4)], True, 12, "random", 12, "random"], # Case 4
    ]
)
def test_Initializers__initialize_random_start_functional(train_X, train_Y, covars, random_start, num_initial_random,
                                                          random_sampling_method, num_initial_random_points_res,
                                                          sampling_method_res):
    """
    test that initialization of random start works. There are 4 cases to consider (case -> expected behavior)
    - CASE 1: train_X, train_Y present; random_start = False -> (self.num_initial_random_points = 0, self.random_sampling_method = None)
    - CASE 2: train_X, train_Y present; random_start = True -> (self.num_initial_random_points = user-provided number / round(sqrt(num_covariates)), self.random_sampling_method = user-provided)
    - CASE 3: train_X, train_Y NOT present; random_start = False -> (this is a case of INCONSISTENCY in user input. Expect user has made a mistake so proceeds but throws warning. self.num_initial_random_points = round(sqrt(num_covariates)), self.random_sampling_method = user-provided)
    - CASE 4: train_X, train_Y NOT present; random_start = True -> (self.num_initial_random_points = user-provided number / round(sqrt(num_covariates)), self.random_sampling_method = user-provided)
    """

    # initialize class
    cls = Initializers()

    # set attributes
    cls.train_Y = train_Y
    cls.train_X = train_X
    cls.covars = covars

    # run method
    cls._Initializers__initialize_random_start(random_start=random_start, num_initial_random=num_initial_random,
                                               random_sampling_method=random_sampling_method)

    # assert outcome
    assert cls.num_initial_random_points == num_initial_random_points_res
    assert cls.random_sampling_method == sampling_method_res


@pytest.mark.parametrize(
    "covars, total_num_covars, covar_mapped_names, GP_kernel_mapping_covar_identification, covar_details",
    [
        [[(1, 0, 2)], 1, ["covar0"], [{"type": int, "column": [0]}], {"covar0":{"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}}],
        [[(1, 0, 2), ("hej", "med", "dig", "hr")], 5, ["covar0", "covar1_hej", "covar1_med", "covar1_dig", "covar1_hr"], [{"type": int, "column": [0]}, {"type": str, "column": [1, 2, 3, 4]}], {"covar0":{"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}, "covar1": {"guess": "hej", "options": {"hej", "med", "dig", "hr"}, "type": str, "columns": [1, 2, 3, 4], "opt_names": ["covar1_hej", "covar1_med", "covar1_dig", "covar1_hr"]}}],
    ]
)
def test__initialize_covars_list_of_tuples_integration_works(covars, total_num_covars, covar_mapped_names,
                                                 GP_kernel_mapping_covar_identification, covar_details):
    """
    test that the right attributes 'covar_details', 'GP_kernel_mapping_covar_identification', 'covar_mapped_names' and
    'total_num_covars' are created correctly. Tests for both single and multiple-covariate case, and also
    tests both numerical and categorical covariates
    """

    # initialize class
    cls = Initializers()

    # run the method
    cls._Initializers__initialize_covars_list_of_tuples(covars=covars)

    # assert
    assert cls.total_num_covars == total_num_covars
    assert cls.GP_kernel_mapping_covar_identification == GP_kernel_mapping_covar_identification
    assert cls.covar_mapped_names == covar_mapped_names
    assert cls.covar_details == covar_details


@pytest.mark.parametrize(
    "covars, error_msg",
    [
        [[(1, 0, 2, 3)], "creative_project._validators.__validate_num_entries_covar_tuples: tuple entries of types (int, float) must have 3 entries. This is not the case for the entry (1, 0, 2, 3)"], # test correct number of entries (_validators.__validate_num_entries_covar_tuples)
        [[(1, 0, 2), ()], "creative_project._validators.__validate_num_entries_covar_tuples: tuple entries of types (int, float) must have 3 entries. This is not the case for the entry ()"], # test correct number of entries (_validators.__validate_num_entries_covar_tuples)
        [[(1.1, 0.2, 3.4), (1, True, 2)], "creative_project._initializers.Initialzer.__determine_tuple_datatype: individual covariates provided via tuples can only be of types ('float', 'int', 'str') but was provided <class 'bool'>"], # test correct data types (_validators.__validate_covars)
    ]
)
def test__initialize_covars_list_of_tuples_integration_fails(covars, error_msg):
    """
    test that the right error messages are produced and that the method does not create the attributes 'covar_details',
    'GP_kernel_mapping_covar_identification', 'covar_mapped_names' and 'total_num_covars' if the list of tuples 'covars'
    does not follow required format
    """

    # initialize class
    cls = Initializers()

    # run the method
    with pytest.raises(Exception) as e:
        cls._Initializers__initialize_covars_list_of_tuples(covars=covars)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "covars, total_num_covars, covar_mapped_names, GP_kernel_mapping_covar_identification, covar_details",
    [
        [{"var0": {"guess":1, "min": 0, "max": 2, "type": int}}, 1, ["var0"], [{"type": int, "column": [0]}], {"var0":{"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}}],
        [{"var0": {"guess":1, "min": 0, "max": 2, "type": int}, "idiot": {"guess": "hej", "options": {"hej", "med", "dig", "hr"}, "type": str}}, 5, ["var0", "idiot_hej", "idiot_med", "idiot_dig", "idiot_hr"], [{"type": int, "column": [0]}, {"type": str, "column": [1, 2, 3, 4]}], {"var0": {"guess":1, "min": 0, "max": 2, "type": int, "columns": 0}, "idiot": {"guess": "hej", "options": {"hej", "med", "dig", "hr"}, "type": str, "columns": [1, 2, 3, 4], "opt_names": ["idiot_hej", "idiot_med", "idiot_dig", "idiot_hr"]}}],
    ]
)
def test__initialize_covars_dict_of_dicts_integration_works(covars, total_num_covars, covar_mapped_names,
                                                GP_kernel_mapping_covar_identification, covar_details):
    """
    test that the right attributes 'covar_details', 'GP_kernel_mapping_covar_identification', 'covar_mapped_names' and
    'total_num_covars' are created correctly. Tests for both single and multiple-covariate case, and also
    tests both numerical and categorical covariates.

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
        [{'var0': {'guess': 1, 'min': 0, 'max': 2}, 'var1': (1, 0, 3)}, "creative_project._validators.Validators.__validate_covars_dict_of_dicts: 'covars' provided as part of class initialization must be either a list of tuples or a dict of dicts. Current provided is a dict containing data types {<class 'dict'>, <class 'tuple'>}."],  # incorrect data type in 'covars'
        [{'var0': {'guess': 1, 'min': 0, 'max': 2}}, "creative_project._validators.Validators.__validate_covars_dict_of_dicts: key 'type' missing for covariate 'var0' (covars['var0']={'guess': 1, 'min': 0, 'max': 2})."], # should fail for not having element "type"
        [{'var0': {'guess': 1, 'max': 2, 'type': int}}, "creative_project._validators.Validators.__validate_covars_dict_of_dicts: key 'min' missing for covariate 'var0' (covars['var0']={'guess': 1, 'max': 2, 'type': <class 'int'>})."], # should fail for missing 'min'
        [{'var0': {'guess': 1, 'min': 0, 'type': int}}, "creative_project._validators.Validators.__validate_covars_dict_of_dicts: key 'max' missing for covariate 'var0' (covars['var0']={'guess': 1, 'min': 0, 'type': <class 'int'>})."], # should fail for missing 'max'
        [{'var0': {'guess': 1, 'max': 2, 'type': int}, 'var1': {'guess': 'red', 'options':{'red', 'green'}, 'type': str}}, "creative_project._validators.Validators.__validate_covars_dict_of_dicts: key 'min' missing for covariate 'var0' (covars['var0']={'guess': 1, 'max': 2, 'type': <class 'int'>})."], # should fail for missing 'min' even though more variables provided
        [{'var0': {'guess': 'red', 'type': str}}, "creative_project._validators.Validators.__validate_covars_dict_of_dicts: key 'options' missing for covariate 'var0' (covars['var0']={'guess': 'red', 'type': <class 'str'>})."],  # missing 'guess"
    ]
)
def test__initialize_covars_dict_of_dicts_integration_fails(covars, error_msg):
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