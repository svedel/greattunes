from creative_project._initializers import Initializers
from creative_project._best_response import _find_max_response_value
import pytest
import torch


def test_Initializers__initialize_best_response_functional(custom_models_simple_training_data_4elements,
                                                           tmp_Initializers_with_find_max_response_value_class):
    """
    test initialization of best response data structures based on input data
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

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

    # define required attributes for test to pass (IRL set in CreativeProject which is a child of Initializers)
    cls.dtype = torch.double
    cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run the function
    cls._Initializers__initialize_best_response()

    # assert that the function has been applied
    assert isinstance(cls.covars_best_response_value, torch.Tensor)
    assert isinstance(cls.best_response_value, torch.Tensor)

    # check size
    assert cls.covars_best_response_value.shape[0] == train_X.shape[0]
    assert cls.best_response_value.shape[0] == train_Y.shape[0]

    # check values, compare to mock_find_max_response_value
    # below two "test"-tensors contain best values corresponding to data from
    # conftest.py's "custom_models_simple_training_data_4elements"
    test_max_covars_test = torch.tensor([[-1.0], [-1.0], [-0.5], [1.0]], dtype=torch.double)
    test_max_response_test = torch.tensor([[0.2], [0.2], [0.5], [2.0]], dtype=torch.double)
    for it in range(train_X.shape[0]):
        assert cls.covars_best_response_value[it].item() == test_max_covars_test[it].item()
        assert cls.best_response_value[it].item() == test_max_response_test[it].item()


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