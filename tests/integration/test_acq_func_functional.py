import pytest
import torch
from botorch.acquisition import ExpectedImprovement
from greattunes._acq_func import AcqFunction
import greattunes._acq_func


def test_AcqFunction__initialize_acq_func_functional(custom_models_simple_training_data_4elements, monkeypatch):
    """
    test the private method __initialize_acq_func. This does a data validation check on self.train:Y and then calls
    AcqFunction.set_acq_func; monkeypatch this dependency
    """

    # get the data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    cls = AcqFunction()
    cls.it_ran = False
    cls.train_Y = train_Y

    # monkeypatching
    def mock_set_acq_func(self):
        self.it_ran = True
        return True
    monkeypatch.setattr(
        creative_project._acq_func.AcqFunction, "set_acq_func", mock_set_acq_func
    )

    # test that it passes with attribute train_Y being not None (should return True)
    cls._AcqFunction__initialize_acq_func()
    assert cls.it_ran

    # test that it fails with attribute train_Y being set to None
    cls.train_Y = None
    cls.it_ran = False
    cls._AcqFunction__initialize_acq_func()
    assert not cls.it_ran


def test_acq_func_identify_new_candidate_withdatacount_functional(covars_for_custom_models_simple_training_data_4elements,
                                              ref_model_and_training_data):
    """
    test that acquisition function optimization works. Test when some training data iterations taken (i.e.
    self.model["covars_sampled_iter"] > 0)
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

    # set model attributes needed for test: train_Y to not None, model to None
    cls.train_Y = train_Y
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.model = {"model": model_obj,
                 "likelihood": lh,
                 "loglikelihood": ll,
                 "covars_proposed_iter": train_X.shape[0],
                 "covars_sampled_iter": train_X.shape[0],
                 "response_sampled_iter": train_Y.shape[0],
                 }

    cls.acq_func = {
        "type": "EI",  # define the type of acquisition function
        "object": ExpectedImprovement(model=cls.model["model"], best_f=train_Y.max().item())
    }

    # set covariate attributes needed for test
    cls.initial_guess = torch.tensor([[g[0]] for g in covars], dtype=torch.double,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cls.covar_bounds = torch.tensor([[g[1] for g in covars], [g[2] for g in covars]], dtype=torch.double,
                                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cls.num_initial_random_points = 0
    cls.random_step_cadence = 10

    # run test
    candidate = cls.identify_new_candidate()

    # assert
    assert round(candidate[0].item(), 3) == 0.706


def test_acq_func_identify_new_candidate_withdatacount_multivariate_functional(ref_model_and_multivariate_training_data):
    """
    test that acquisition function optimization works. Test when some training data iterations taken (i.e.
    self.model["covars_sampled_iter"] > 0)
    """

    # load data
    covars = ref_model_and_multivariate_training_data[0]
    train_X = ref_model_and_multivariate_training_data[1]
    train_Y = ref_model_and_multivariate_training_data[2]

    # load pretrained model
    model_obj = ref_model_and_multivariate_training_data[3]
    lh = ref_model_and_multivariate_training_data[4]
    ll = ref_model_and_multivariate_training_data[5]

    # define class instance, set appropriate attributes
    # the acq func
    cls = AcqFunction()

    # set model attributes needed for test: train_Y to not None, model to None
    cls.train_Y = train_Y
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.model = {"model": model_obj,
                 "likelihood": lh,
                 "loglikelihood": ll,
                 "covars_proposed_iter": train_X.shape[0],
                 "covars_sampled_iter": train_X.shape[0],
                 "response_sampled_iter": train_Y.shape[0],
                 }

    cls.acq_func = {
        "type": "EI",  # define the type of acquisition function
        "object": ExpectedImprovement(model=cls.model["model"], best_f=train_Y.max().item())
    }

    # set covariate attributes needed for test
    cls.initial_guess = torch.tensor([[g[0]] for g in covars], dtype=torch.double,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cls.covar_bounds = torch.tensor([[g[1] for g in covars], [g[2] for g in covars]], dtype=torch.double,
                                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cls.num_initial_random_points = 0
    cls.random_step_cadence = 10

    # run test
    candidate = cls.identify_new_candidate()

    # assert
    candidate_results = torch.tensor([[ 0.1121, 11.6469, -2.8049]], dtype=torch.double,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for it in range(candidate.shape[1]):
        assert round(candidate[0, it].item(),4) == candidate_results[0, it].item()


@pytest.mark.parametrize(
    "covars_sampled_iter, num_initial_random_points, random_sampling_method",
    [
        [0, 3, "random"],  # initial random period
        [8, 3, "latin_hcs"],  # interdispersed random point
    ]
)
def test_AcqFunction_random_candidate_int_works(covars_sampled_iter, num_initial_random_points, random_sampling_method):
    """
    test that 'random_candidate' works for both initial iterations (generate during first call), as well as for
    later interdispersed random datapoints.

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
