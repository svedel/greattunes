import pytest
import botorch
import torch
from greattunes._acq_func import AcqFunction
from greattunes.utils import DataSamplers

def test_acq_func_set_acq_func_fails(custom_models_simple_training_data_4elements):
    """
    test that set_acq_func fails if either model or train_Y are not set
    """

    # get the data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # the acq func
    cls = AcqFunction()
    cls.acq_func = {
        "type": "ExpectedImprovement", # define the type of acquisition function
        "object": None
    }

    # set attributes needed for test: train_Y to not None, model to None
    cls.train_Y = train_Y
    cls.model = {"model": None}

    with pytest.raises(Exception) as e:
        assert cls.set_acq_func()
    assert str(e.value) == "greattunes.greattunes._acq_func.AcqFunction.set_acq_func: no surrogate model set " \
                           "(self.model['model'] is None)"

    # set attributes needed for test: train_Y to not None, model to "something" (something that doesn't trigger exception)
    cls.train_Y = None
    cls.model = {"model": "something"}

    with pytest.raises(Exception) as e:
        assert cls.set_acq_func()
    assert str(e.value) == "greattunes.greattunes._acq_func.AcqFunction.set_acq_func: no training data provided " \
                           "(self.train_Y is None)"

    # set attributes needed for test: train_Y to not None, model to None. Model exception should fire first
    cls.train_Y = None
    cls.model = {"model": None}

    with pytest.raises(Exception) as e:
        assert cls.set_acq_func()
    assert str(e.value) == "greattunes.greattunes._acq_func.AcqFunction.set_acq_func: no surrogate model set " \
                           "(self.model['model'] is None)"

def test_acq_func_set_acq_func_fails_wrong_acqfunc_name(ref_model_and_training_data):
    """
    test that set_acq_func does not set acquisition function if wrong name chosen
    """

    # load data and model
    train_X = ref_model_and_training_data[0]
    train_Y = ref_model_and_training_data[1]

    # load pretrained model
    model_obj = ref_model_and_training_data[2]
    lh = ref_model_and_training_data[3]
    ll = ref_model_and_training_data[4]

    # the acq func
    cls = AcqFunction()
    cls.acq_func = {
        "type": "WRONG",  # define the type of acquisition function
        "object": None
    }

    # set attributes needed for test: train_Y to not None, model to None
    cls.train_Y = train_Y
    cls.model = {"model": model_obj,
                 "likelihood": lh,
                 "loglikelihood": ll,
                 }

    with pytest.raises(Exception) as e:
        assert cls.set_acq_func()
    assert str(e.value) == "greattunes.greattunes._acq_func.AcqFunction.set_acq_func: unsupported acquisition " \
                           "function name provided. '" + cls.acq_func["type"] + "' not in list of supported " \
                           "acquisition functions [AnalyticAcquisitionFunction, ConstrainedExpectedImprovement, " \
                            "ConstrainedMCObjective, ExpectedImprovement, MCAcquisitionObjective, " \
                            "NoisyExpectedImprovement, PosteriorMean, ProbabilityOfImprovement, UpperConfidenceBound, "\
                            "qExpectedImprovement, qKnowledgeGradient, qMaxValueEntropy, " \
                            "qMultiFidelityMaxValueEntropy, qNoisyExpectedImprovement, qProbabilityOfImprovement, " \
                            "qSimpleRegret, qUpperConfidenceBound]."


def test_acq_func_set_acq_func_works(ref_model_and_training_data):
    """
    test that set_acq_func works with model and training data are provided correctly
    """

    # load data and model
    train_X = ref_model_and_training_data[0]
    train_Y = ref_model_and_training_data[1]

    # load pretrained model
    model_obj = ref_model_and_training_data[2]
    lh = ref_model_and_training_data[3]
    ll = ref_model_and_training_data[4]

    # the acq func
    cls = AcqFunction()
    cls.acq_func = {
        "type": "ExpectedImprovement",  # define the type of acquisition function
        "object": None
    }

    # set attributes needed for test: train_Y to not None, model to None
    cls.train_Y = train_Y
    cls.model = {"model": model_obj,
                 "likelihood": lh,
                 "loglikelihood": ll,
                 }

    # set the acquisition function
    cls.set_acq_func()

    assert cls.acq_func["object"] is not None
    assert isinstance(cls.acq_func["object"], botorch.acquisition.acquisition.AcquisitionFunction)



def test_acq_func_identify_new_candidate_nodatacount_unit(covars_for_custom_models_simple_training_data_4elements,
                                              ref_model_and_training_data, monkeypatch):
    """
    test that acquisition function optimization works. Test when no training data iterations taken (i.e.
    self.model["covars_sampled_iter"] == 0). Monkeypatch method 'random_candidate'
    """

    # load data
    covars = covars_for_custom_models_simple_training_data_4elements
    train_X = ref_model_and_training_data[0]
    train_Y = ref_model_and_training_data[1]

    # load pretrained model
    model_obj = ref_model_and_training_data[2]
    lh = ref_model_and_training_data[3]
    ll = ref_model_and_training_data[4]

    # define class instance, set appropriate attributes
    # the acq func
    cls = AcqFunction()
    cls.acq_func = {
        "type": "ExpectedImprovement",  # define the type of acquisition function
        "object": None
    }

    # set model attributes needed for test: train_Y to not None, model to None
    cls.train_Y = train_Y
    cls.train_X = train_X
    cls.model = {"model": model_obj,
                 "likelihood": lh,
                 "loglikelihood": ll,
                 "covars_proposed_iter": 0,
                 "covars_sampled_iter": 0,
                 "response_sampled_iter": 0,
                 }

    # set covariate attributes needed for test
    init_guess = torch.tensor([[g[0]] for g in covars], dtype=torch.double,
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cls.initial_guess = init_guess
    cls.num_initial_random_points = 1
    cls.random_step_cadence = 10

    # monkeypatch AcqFunction.random_candidate
    def mock_random_candidate():
        return init_guess
    monkeypatch.setattr(cls, "random_candidate", mock_random_candidate)

    # run test
    candidate = cls.identify_new_candidate()

    # assert
    for it in range(len(covars)):
        assert candidate[it].item() == covars[it][0]


@pytest.mark.parametrize(
    "train_X, train_Y",
    [
        [torch.tensor([[1.1]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double)],
        [torch.tensor([[1.1, -4.234, 7.65]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double)],
        [None, None]
]
)
def test__initialize_acq_func(train_X, train_Y, monkeypatch):
    """
    test that "__initialize_acq_func" works by testing the two scenarios that influence the outcome: having train_Y
    being None or not. Monkeypatching "set_acq_func" in "__initialize_acq_func".
    """

    ### First test: iterations have already started (no initialization accepted)
    # initialize
    cls = AcqFunction()
    cls.train_X = train_X
    cls.train_Y = train_Y

    # set counters. In this test have made sure that train_X, train_Y are either None, None or not-None, not-None
    if (train_X is not None) and (train_Y is not None):
        cls.model = {
            "covars_sampled_iter": 1,
            "response_sampled_iter": 1
        }
    else:
        cls.model = {
            "covars_sampled_iter": 0,
            "response_sampled_iter": 0
        }

    # attribute to store acq function
    cls.acq_func = {"object": None}

    # monkeypatch "set_acq_func"
    set_string = "set"
    def mock_set_acq_func():
        cls.acq_func["object"] = set_string

    monkeypatch.setattr(cls, "set_acq_func", mock_set_acq_func)

    # run the method
    cls._AcqFunction__initialize_acq_func()

    # assert
    if train_X is None:
        assert cls.acq_func["object"] is None
    else:
        assert cls.acq_func["object"] == set_string


@pytest.mark.parametrize(
    "sampled_iter, num_initial_random_points, random_step_cadence, sample_type_result",
    [
        [1, 3, 5, "random"], # in initial random segment
        [2, 3, 5, "random"], # last in initial random segment
        [3, 3, 5, "bayesian"], # first bayesian
        [7, 3, 5, "random"], # first interdispersed random
        [8, 3, 5, "bayesian"], # first point after first interdispersed random --- this should be bayesian
]
)
def test__random_or_bayesian_unit(sampled_iter, num_initial_random_points, random_step_cadence, sample_type_result):
    """
    test that '__random_or_bayesian' works. See docstring for method for details of intended behavior
    """

    # initialize class
    cls = AcqFunction()

    # set attributes
    cls.num_initial_random_points = num_initial_random_points
    cls.random_step_cadence = random_step_cadence
    cls.model = {"covars_sampled_iter": sampled_iter}

    # run method
    sample_type = cls._AcqFunction__random_or_bayesian()

    # assert that the expected result is returned
    assert sample_type == sample_type_result


@pytest.mark.parametrize(
    "covars_sampled_iter, num_initial_random_points, random_sampling_method",
    [
        [0, 3, "random"],  # initial random period
        [8, 3, "latin_hcs"],  # interdispersed random point
    ]
)
def test_AcqFunction_random_candidate_works(covars_sampled_iter, num_initial_random_points, random_sampling_method, monkeypatch):
    """
    test that 'random_candidate' works for both initial iterations (generate during first call), as well as for
    later interdispersed random datapoints. Monkeypatch both methods from utils.DataSamplers

    Note that we only test the initial random datapoints starting at the first iteration. That's because the list of
    random candidates for datapoints is only generated during this first step
    """

    # initialize class
    cls = AcqFunction()

    # set attributes
    cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls.covar_bounds = torch.tensor([[0, 1, 1.5],[1.3, 1.8, 3.7]], dtype=torch.double, device=cls.device)
    cls.initial_guess = torch.tensor([[0.6, 1.5, 2.2]], dtype=torch.double, device=cls.device)
    cls.model = {"covars_sampled_iter": covars_sampled_iter}
    cls.num_initial_random_points = num_initial_random_points
    cls.random_sampling_method = random_sampling_method

    # monkeypatch
    def mock_random(n_samp, initial_guess, covar_bounds, device):
        candidates = torch.rand((n_samp, cls.initial_guess.shape[1]), dtype=torch.double, device=cls.device)
        return candidates
    monkeypatch.setattr(DataSamplers, "random", mock_random)
    monkeypatch.setattr(DataSamplers, "latin_hcs", mock_random)

    # run method
    candidate = cls.random_candidate()

    # assert only one candidate
    assert candidate.size()[0] == 1

    # assert correct number of rows
    assert candidate.size()[1] == cls.initial_guess.shape[1]

    # assert that candidates torch tensor has been stored as attribute
    if cls.model["covars_sampled_iter"] == 0:
        assert hasattr(cls, "_AcqFunction__initial_random_candidates")
        assert cls._AcqFunction__initial_random_candidates.size()[0] == num_initial_random_points
        assert cls._AcqFunction__initial_random_candidates.size()[1] == cls.initial_guess.shape[1]
