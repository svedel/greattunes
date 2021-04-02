from botorch.models import SingleTaskGP
import numpy as np
import pytest
import random
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from creative_project._initializers import Initializers
from creative_project._validators import Validators

### Parsing of keywords: allow for specialized tests for different python versions
def pytest_addoption(parser):
    parser.addoption("--pythontestvers", action="store", default="3.8")

@pytest.fixture(autouse=True)
def pythontestvers(request):
    return request.config.option.pythontestvers


### Fixing state of random number generators for test reproducibility
@pytest.fixture(autouse=True)
def rng_state_tests():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


### Simple training data
@pytest.fixture(scope="class")
def custom_models_simple_training_data():
    """
    defines very simple dataset for training of custom GP models. Defined in torch.tensor format
    :return: train_X (torch.tensor)
    :return: train_Y (torch.tensor)
    """
    train_X = torch.tensor([[-1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.2]], dtype=torch.double)
    return train_X, train_Y


@pytest.fixture(scope="class")
def custom_models_simple_training_data_4elements():
    """
    defines very simple dataset for training of custom GP models. Defined in torch.tensor format
    :return: train_X (torch.tensor)
    :return: train_Y (torch.tensor)
    """
    train_X = torch.tensor([[-1.0], [-1.1], [-0.5], [1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.2], [0.15], [0.5], [2.0]], dtype=torch.double)
    return train_X, train_Y

@pytest.fixture(scope="class")
def custom_models_simple_training_data_4elements_covar_details():
    """
    defines covar_details and covar_mapped_names that work with custom_models_simple_training_data_4elements
    """

    covar_details = {"covar0": {"guess": 0.0, "min": -2.0, "max": 2.0, "type": float, "columns": 0}}
    covar_mapped_names = ["covar0"]
    GP_kernel_mapping_covar_identification = [{"type": float, "column": [0]}]
    return covar_details, covar_mapped_names, GP_kernel_mapping_covar_identification


@pytest.fixture(scope="class")
def covars_for_custom_models_simple_training_data_4elements():
    """
    defines initial covars compatible with custom_models_simple_training_data_4elements above
    :return: covars (list of tuple)
    """
    covars = [(0.0, -2.0, 2.0)]
    return covars

@pytest.fixture(scope="class")
def covar_details_covars_for_custom_models_simple_training_data_4elements():
    """
    covar_details corresponding to the covars in covars_for_custom_models_simple_training_data_4elements
    """

    covar_details = {"covar0": {"guess": 0.0, "min": -2.0, "max": 2.0, "type": int, "columns": 0}}
    covar_mapped_names = ["covar0"]
    return covar_details, covar_mapped_names


@pytest.fixture(scope="class")
def covars_initialization_data():
    """
    defines simple and more complex initial covariate datasets to test initialization method
    (._initializers.Initializers__initialize_from_covars)
    :return: covar_simple, covar_complex (lists of tuples of doubles)
    """

    covar_simple = [(0.5, 0, 1)]
    covar_complex = [(0.5, 0, 1), (12.5, 8, 15), (-2, -4, 1.1)]
    return covar_simple, covar_complex


@pytest.fixture(scope="class")
def training_data_covar_complex(covars_initialization_data):
    """
    defines simple training data that corresponds to covar_complex (covars_initialization_data[1]), where covar_complex
    is the right format for initialization of the full user-facing class CreativeProject
    (creative_project.CreativeProject)
    """

    covars = covars_initialization_data[1]

    # the covar training data: building it by taking the covars and in each row adding the factor from the y vector
    train_X = torch.tensor([[x[0]+y for x in covars] for y in [0, -0.5, 1.2]], dtype=torch.double)
    train_Y = torch.tensor([[1.1], [5.5], [0.1]], dtype=torch.double)

    # the covars initialization data
    covar_details = {}
    covar_mapped_names = []
    GP_kernel_mapping_covar_identification = []
    for i in range(len(covars)):
        name = "covar" + str(i)
        covar_details["name"] = {"guess": covars[i][0], "min": covars[i][1], "max": covars[i][2], "type": float, "columns": i}
        covar_mapped_names += [name]
        GP_kernel_mapping_covar_identification += [{"type": float, "column": [i]}]

    return covars, train_X, train_Y, covar_details, covar_mapped_names, GP_kernel_mapping_covar_identification


### Trained GP model
@pytest.fixture(scope="class")
def ref_model_and_training_data(custom_models_simple_training_data_4elements):
    """
    defines a simple, univariate GP model and the data it is defined by
    :return: train_X, train_Y (training data, from custom_models_simple_training_data_4elements above)
    :return: model_obj (model object, SingleTaskGP)
    :return: lh, ll (model likelihood and marginal log-likelihood)
    """

    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # set up the model
    model_obj = SingleTaskGP(train_X, train_Y)

    # the likelihood
    lh = model_obj.likelihood

    # define the "loss" function
    ll = ExactMarginalLogLikelihood(lh, model_obj)

    return train_X, train_Y, model_obj, lh, ll


@pytest.fixture(scope="class")
def ref_model_and_multivariate_training_data(training_data_covar_complex):
    """
    defines a multivariate GP model and the data it is defined by
    :return: covars
    :return: train_X, train_Y (training data, from custom_models_simple_training_data_4elements above)
    :return: model_obj (model object, SingleTaskGP)
    :return: lh, ll (model likelihood and marginal log-likelihood)
    """

    covars = training_data_covar_complex[0]
    train_X = training_data_covar_complex[1]
    train_Y = training_data_covar_complex[2]

    # set up the model
    model_obj = SingleTaskGP(train_X, train_Y)

    # the likelihood
    lh = model_obj.likelihood

    # define the "loss" function
    ll = ExactMarginalLogLikelihood(lh, model_obj)

    return covars, train_X, train_Y, model_obj, lh, ll

### initiated classes for testing
@pytest.fixture(scope="module")
def tmp_observe_class():
    """
    temporary class to allow testing of methods from creative_project._observe
    """

    # define class
    class TmpClass(Validators):
        def __init__(self):
            self.sampling = {"method": None,
                             "response_func": None}
            self.dtype = torch.double
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from creative_project._observe import _get_and_verify_response_input, _get_response_function_input, \
            _read_response_manual_input, _print_candidate_to_prompt, _read_covars_manual_input, \
            _get_and_verify_covars_input, _get_covars_datapoint, _get_response_datapoint

    cls = TmpClass()

    return cls


@pytest.fixture(scope="module")
def tmp_modeling_class():
    """
    temporary class to allow testing of methods from creative_project._modeling
    """

    class TmpClass:
        def __init__(self):
            self.train_X = None
            self.proposed_X = None
            self.train_Y = None
            self.x_data = None
            self.y_data = None

            self.model = {"model_type": None,
                          "likelihood": None,
                          "loglikelihood": None,
                          "response_sampled_iter": 0
                          }

        # import method
        from creative_project._modeling import _set_GP_model

    # initialize class
    cls = TmpClass()

    return cls

@pytest.fixture(scope="function")
def tmp_best_response_class():
    """
    temporary class to test methods stored in _best_response.py
    """

    # test class
    class TmpClass:
        def __init__(self):
            self.train_X = None
            self.train_Y = None
            self.proposed_X = None

            self.covars_best_response_value = None
            self.best_response_value = None
            self.covars_best_response = None
            self.best_response = None

        # import methods
        from creative_project._best_response import _find_max_response_value, _update_max_response_value, \
            current_best, _update_proposed_data

    cls = TmpClass()

    return cls


@pytest.fixture(scope="module")
def tmp_Initializers_with_find_max_response_value_class():
    """
    test version of Initializers to endow it with the property from _find_max_response_value, which is
    otherwise defined as a static method in ._best_response
    """

    class TmpClass(Initializers):
        from creative_project._best_response import _find_max_response_value

    cls = TmpClass()

    return cls


@pytest.fixture(scope="module")
def covar_details_covar_mapped_names():
    """
    examples of matching 'covar_details' and 'covar_mapped_names' for a case of the following variables
    - a: int
    - b: float
    - c: categorical (str), with options "red", "blue" and "green"
    """

    covar_details = \
        {
            'a': {
                'guess': 1,
                'min': -1,
                'max': 3,
                'type': int,
                'columns': 0,
                },
            'b': {
                'guess': 2.2,
                'min': -1.7,
                'max': 4.2,
                'type': float,
                'columns': 1,
            },
            'c': {
                'guess': 'red',
                'options': {'red', 'green', 'blue'},
                'type': str,
                'columns': [2, 3, 4],
                'opt_names': ['c_red', 'c_green', 'c_blue'],
            }
        }

    covar_mapped_names = ['a', 'b', 'c_red', 'c_green', 'c_blue']

    return covar_details, covar_mapped_names


@pytest.fixture(scope="module")
def covar_details_mapped_covar_mapped_names_tmp_observe_class():
    """
    covar_details and covar_mapped_names corresponding to the sample problems being tested by tmp_observe_class in
    tests/unit/test_observe_unit.py
    """

    covar_details = {
        "covar0": {
            "guess": 0.1,
            "min": -1.0,
            "max": 2.0,
            "type": float,
            "columns": 0,
            "pandas_column": 0,
        },
        "covar1": {
            "guess": 2.5,
            "min": -1.0,
            "max": 3.0,
            "type": float,
            "columns": 1,
            "pandas_column": 1,
        },
        "covar2": {
            "guess": 12,
            "min": 0,
            "max": 250,
            "type": float,
            "columns": 2,
            "pandas_column": 2,
        },
        "covar3": {
            "guess": 0.22,
            "min": -2.0,
            "max": 1.0,
            "type": float,
            "columns": 3,
            "pandas_column": 3,
        },
    }

    covar_mapped_names = ["covar0", "covar1", "covar2", "covar3"]
    sorted_pandas_columns = ["covar0", "covar1", "covar2", "covar3"]

    return covar_details, covar_mapped_names, sorted_pandas_columns