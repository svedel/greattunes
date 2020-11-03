import pytest
import torch
from botorch.acquisition import ExpectedImprovement
from creative_project._acq_func import AcqFunction
import creative_project._acq_func


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
    cls.start_from_guess = False

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
    cls.start_from_guess = False

    # run test
    candidate = cls.identify_new_candidate()

    # assert
    candidate_results = torch.tensor([[ 0.1121, 11.6469, -2.8049]], dtype=torch.double,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for it in range(candidate.shape[1]):
        assert round(candidate[0, it].item(),4) == candidate_results[0, it].item()

