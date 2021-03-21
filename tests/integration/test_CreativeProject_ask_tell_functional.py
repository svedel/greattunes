import functools
import pytest
import torch
from creative_project import CreativeProject


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter, random_start",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, True],  # case 1 with random start
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, True],  # case 2 with random start
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, False]
    ]
)
def test_CreativeProject_ask_integration_test_works(covars, model_type, train_X, train_Y, covars_proposed_iter,
                                                    covars_sampled_iter, response_sampled_iter, random_start):
    """
    test the positive cases for CreativeProject.ask method.
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=random_start)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter
    if covars_proposed_iter > 0:
        cc.num_initial_random_points = 0
        cc.random_sampling_method = None

        # run the method
    cc.ask()

    # check that an entry has been added to cls.proposed_X
    if train_Y is not None:
        assert cc.proposed_X.size()[0] == train_X.size()[0] + 1
        assert cc.proposed_X[-1].size()[0] == train_X.size()[1]  # check that the new candidate has the right number of entries
    else:
        assert cc.proposed_X.size()[0] == 1

    # check that a model and an acquisition function have been assigned if starting from no data (train_X, train_Y is None)
    if train_Y is not None:
        assert cc.acq_func["object"] is not None
        assert cc.model["model"] is not None

    # assert that the number of covars in returned "proposed_X" matches the number from "covars"
    assert cc.proposed_X.size()[1] == len(covars)

    # check that the counter 'covars_proposed_iter' is updated
    assert cc.model["covars_proposed_iter"] == covars_proposed_iter +1


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter, error_msg",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, torch.tensor([[0.8]], dtype=torch.double), 0, 0, 0, "kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: no surrogate model set (self.model['model'] is None)"],  # the case where no COVARIATE data is available
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, torch.tensor([[0.8], [22]], dtype=torch.double), 0, 0, 0, "kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: no surrogate model set (self.model['model'] is None)"],  # the case where no COVARIATE data is available, multiple observations
    ]
)
def test_CreativeProject_ask_integration_test_fails(covars, model_type, train_X, train_Y, covars_proposed_iter,
                                                    covars_sampled_iter, response_sampled_iter, error_msg):
    """
    test the negative cases for CreativeProject.ask method. Currently testing the case where only train_Y data has been
    added, not train_X data
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter

    # run the method
    with pytest.raises(Exception) as e:
        cc.ask()
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 2, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 2, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 2, 1, 1]
    ]
)
def test_CreativeProject_tell_integration_test_works(covars, model_type, train_X, train_Y, covars_proposed_iter,
                                                     covars_sampled_iter, response_sampled_iter, monkeypatch):
    """
    test the positive cases for CreativeProject.tell method. Monkeypatch "_read_covars_manual_input" and
    "_read_response_manual_input" from ._observe.py to circumvent manual input via builtins.input
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)
    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor
    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)
    def mock_read_response_manual_input(additional_text):
        return resp_tensor
    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the method
    cc.tell()

    # assert that a model has been added (start from cc.model["model"] = None)
    assert cc.model["model"] is not None

    # assert that a new observation has been added for covariates
    if train_X is not None:
        assert cc.train_X.size()[0] == train_X.size()[0] + 1
    else:
        assert cc.train_X.size()[0] == 1

    # assert that the right elements have been added to the covariate observation
    for i in range(cc.train_X.size()[1]):
        assert cc.train_X[-1, i].item() == candidate_tensor[0, i].item()

    # assert that a new observation has been added for the response
    if train_Y is not None:
        assert cc.train_Y.size()[0] == train_Y.size()[0] + 1
    else:
        assert cc.train_Y.size()[1] == 1

    # assert that the right elements have been added to the response observation
    assert cc.train_Y[-1, 0].item() == resp_tensor[0, 0].item()


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 1, 0, 0],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 1, 0, 0],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1]
    ]
)
def test_CreativeProject_tell_integration_test_works_overwrite(covars, model_type, train_X, train_Y, covars_proposed_iter,
                                                     covars_sampled_iter, response_sampled_iter, monkeypatch):
    """
    test the positive case for CreativeProject.tell method where last datapoint is overwritten (controlled by
    covars_proposed_iter <= covars_sampled_iter). Monkeypatch "_read_covars_manual_input" and
    "_read_response_manual_input" from ._observe.py to circumvent manual input via builtins.input
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)
    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor
    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)
    def mock_read_response_manual_input(additional_text):
        return resp_tensor
    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the method
    cc.tell()

    # assert that a model has been added (start from cc.model["model"] = None)
    assert cc.model["model"] is not None

    # assert that NO new observation has been added for covariates
    if train_X is not None:
        assert cc.train_X.size()[0] == train_X.size()[0]
    else:
        assert cc.train_X.size()[0] == 1

    # assert that the right elements have been added to the covariate observation (last row overwritten)
    for i in range(cc.train_X.size()[1]):
        assert cc.train_X[-1, i].item() == candidate_tensor[0, i].item()

    # assert that NO new observation has been added for the response
    if train_Y is not None:
        assert cc.train_Y.size()[0] == train_Y.size()[0]
    else:
        assert cc.train_Y.size()[1] == 1

    # assert that the right elements have been added to the response observation (last row overwritten)
    assert cc.train_Y[-1, 0].item() == resp_tensor[0, 0].item()


# add a test that looks for failures in tell (e.g. incorrect added input: make sure nothing is updated)
@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter, covars_cand, resp_cand, error_msg",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1, 2]], dtype=torch.double), torch.tensor([[23]], dtype=torch.double), "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like 'tensor([0.8000], dtype=torch.float64)', but got 'tensor([[1., 2.]], dtype=torch.float64)'"],  # fail on train_X
        [[(1, 0.5, 1.5)], "Custom", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1, 2]], dtype=torch.double), torch.tensor([[23]], dtype=torch.double), "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like 'tensor([0.8000], dtype=torch.float64)', but got 'tensor([[1., 2.]], dtype=torch.float64)'"],  # fail on train_X
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1]], dtype=torch.double), torch.tensor([[23, 11]], dtype=torch.double), "creative_project._observe._get_and_verify_response_input: incorrect number of variables provided. Was expecting input of size (1,1) but received torch.Size([1, 2])"],  # fail on train_Y
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1, 2]], dtype=torch.double), torch.tensor([[23]], dtype=torch.double), "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like 'tensor([  0.8000,   0.2000, 102.0000], dtype=torch.float64)', but got 'tensor([[1., 2.]], dtype=torch.float64)'"],  # fail on train_X, too few entries
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1, 2, 3]], dtype=torch.double), torch.tensor([], dtype=torch.double), "creative_project._observe._get_and_verify_response_input: incorrect number of variables provided. Was expecting input of size (1,1) but received torch.Size([0])"],  # fail on train_Y, too few entries
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1, 2]], dtype=torch.double), torch.tensor([], dtype=torch.double), "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like 'tensor([  0.8000,   0.2000, 102.0000], dtype=torch.float64)', but got 'tensor([[1., 2.]], dtype=torch.float64)'"],  # too few in both train_X and train_Y, fail on train_X since this comes first
    ]
)
def test_CreativeProject_tell_integration_test_fails(covars, model_type, train_X, train_Y, covars_proposed_iter,
                                                     covars_sampled_iter, response_sampled_iter, covars_cand, resp_cand,
                                                     error_msg, monkeypatch):
    """
    test that a failure in "tell" will cause an error and not update any counters. Monkeypatch
    "_read_covars_manual_input" and "_read_response_manual_input" from ._observe.py to circumvent manual input via
    builtins.input
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter

    # original model state
    model_state = cc.model["model"]

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = covars_cand
    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor
    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = resp_cand
    def mock_read_response_manual_input(additional_text):
        return resp_tensor
    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    with pytest.raises(Exception) as e:
        # run the method
        cc.tell()
    assert str(e.value) == error_msg

    # assert that stored train_X is not updated
    assert cc.train_X.size()[0] == train_X.size()[0]
    assert cc.train_X.size()[1] == train_X.size()[1]

    # assert that stored train_Y is not updated
    assert cc.train_Y.size()[0] == train_Y.size()[0]
    assert cc.train_Y.size()[1] == train_Y.size()[1]

    # assert that stored surrogate model is not updated
    assert cc.model["model"] == model_state


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter, random_start",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, True],  # case 1 with random start
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, True],  # case 2 with random start
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, False]
    ]
)
def test_CreativeProject_integration_ask_tell_one_loop_works(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, random_start, monkeypatch):
    """
    test that a single loop of ask/tell works: creates a candidate, creates a model, stores covariates and response.
    Monkeypatch "_read_covars_manual_input" and "_read_response_manual_input" from ._observe.py to circumvent manual
    input via builtins.input. Does not use the kwargs for covariates
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=random_start)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter
    if covars_proposed_iter > 0:
        cc.num_initial_random_points = 0
        cc.random_sampling_method = None

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)

    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor

    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)

    def mock_read_response_manual_input(additional_text):
        return resp_tensor

    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the ask method
    cc.ask()

    # run the tell method
    cc.tell()

    ### assert for ask ###

    # check that an entry has been added to cls.proposed_X
    if train_Y is not None:
        assert cc.proposed_X.size()[0] == train_X.size()[0] + 1
        assert cc.proposed_X[-1].size()[0] == train_X.size()[1]  # check that the new candidate has the right number of entries
    else:
        assert cc.proposed_X.size()[0] == 1

    # assert that the number of covars in returned "proposed_X" matches the number from "covars"
    assert cc.proposed_X.size()[1] == len(covars)

    # check that the counter 'covars_proposed_iter' is updated
    assert cc.model["covars_proposed_iter"] == covars_proposed_iter + 1

    ### check for tell ###

    # assert that a new observation has been added for covariates
    if train_X is not None:
        assert cc.train_X.size()[0] == train_X.size()[0] + 1
    else:
        assert cc.train_X.size()[0] == 1

    # assert that the right elements have been added to the covariate observation
    for i in range(cc.train_X.size()[1]):
        assert cc.train_X[-1, i].item() == candidate_tensor[0, i].item()

    # assert that a new observation has been added for the response
    if train_Y is not None:
        assert cc.train_Y.size()[0] == train_Y.size()[0] + 1
    else:
        assert cc.train_Y.size()[1] == 1

    # assert that the right elements have been added to the response observation
    assert cc.train_Y[-1, 0].item() == resp_tensor[0, 0].item()

    ### check that acquisition function and model have been added

    # check that a model function has been assigned (should happen in all cases as part of tell)
    assert cc.model["model"] is not None

    # check that an acquisition function has been added (only if some data present in train_X, train_Y at first step)
    if train_X is not None:
        assert cc.acq_func["object"] is not None

# need:
# - integration test for kwargs for response, nothing else (positive and negative)
# - integration test for  kwargs for covars and response (positive and negative)


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter, kwarg_covariates, random_start",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, torch.tensor([[1.8]], dtype=torch.double), False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, torch.tensor([[1.8]], dtype=torch.double), False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, torch.tensor([[1.8]], dtype=torch.double), True],  # case 1 with random start
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, torch.tensor([[1.8]], dtype=torch.double), True],  # case 2 with random start
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1.8]], dtype=torch.double), False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[0.8, 0.2, 103]], dtype=torch.double), False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[0.8, 0.2, 103]], dtype=torch.double), False]
    ]
)
def test_CreativeProject_integration_ask_tell_one_loop_kwarg_covars_works(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, kwarg_covariates, random_start,
                                                                          monkeypatch):
    """
    test that a single loop of ask/tell works when providing covars as kwarg to tell: creates a candidate, creates a
    model, stores covariates and response. Monkeypatch "_read_response_manual_input" from ._observe.py to circumvent
    manual input via builtins.input and provides covariates via kwargs
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=random_start)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter
    if covars_proposed_iter > 0:
        cc.num_initial_random_points = 0
        cc.random_sampling_method = None

    # # monkeypatch "_read_covars_manual_input"
    # candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)
    #
    # def mock_read_covars_manual_input(additional_text):
    #     return candidate_tensor
    #
    # monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)

    def mock_read_response_manual_input(additional_text):
        return resp_tensor
    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the ask method
    cc.ask()

    # run the tell method
    cc.tell(covars=kwarg_covariates)


    ### check for tell (no reason to assert for ask)###

    # assert that a new observation has been added for covariates
    if train_X is not None:
        assert cc.train_X.size()[0] == train_X.size()[0] + 1
    else:
        assert cc.train_X.size()[0] == 1

    # assert that the right elements have been added to the covariate observation
    for i in range(cc.train_X.size()[1]):
        assert cc.train_X[-1, i].item() == kwarg_covariates[0, i].item() #candidate_tensor[0, i].item()

    # assert that a new observation has been added for the response
    if train_Y is not None:
        assert cc.train_Y.size()[0] == train_Y.size()[0] + 1
    else:
        assert cc.train_Y.size()[1] == 1

    # assert that the right elements have been added to the response observation
    assert cc.train_Y[-1, 0].item() == resp_tensor[0, 0].item()

    ### check that acquisition function and model have been added

    # check that a model function has been assigned (should happen in all cases as part of tell)
    assert cc.model["model"] is not None

    # check that an acquisition function has been added (only if some data present in train_X, train_Y at first step)
    if train_X is not None:
        assert cc.acq_func["object"] is not None


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter, kwarg_response, random_start",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, torch.tensor([[1.8]], dtype=torch.double), False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, torch.tensor([[1.8]], dtype=torch.double), False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, torch.tensor([[1.8]], dtype=torch.double), True],  # case 1 with random start
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, torch.tensor([[1.8]], dtype=torch.double), True],  # case 2 with random start
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1.8]], dtype=torch.double), False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[103]], dtype=torch.double), False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[0.8]], dtype=torch.double), False]
    ]
)
def test_CreativeProject_integration_ask_tell_one_loop_kwarg_response_works(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, kwarg_response, random_start,
                                                                            monkeypatch):
    """
    test that a single loop of ask/tell works when providing response as kwarg to tell: creates a candidate, creates a
    model, stores covariates and response. Monkeypatch "_read_response_manual_input" from ._observe.py to circumvent
    manual input via builtins.input and provides response via kwargs
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=random_start)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter
    if covars_proposed_iter > 0:
        cc.num_initial_random_points = 0
        cc.random_sampling_method = None

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)
    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor
    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # # monkeypatch "_read_response_manual_input"
    # resp_tensor = torch.tensor([[12]], dtype=torch.double)
    #
    # def mock_read_response_manual_input(additional_text):
    #     return resp_tensor
    # monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the ask method
    cc.ask()

    # run the tell method
    cc.tell(response=kwarg_response)


    ### check for tell (no reason to assert for ask)###

    # assert that a new observation has been added for covariates
    if train_X is not None:
        assert cc.train_X.size()[0] == train_X.size()[0] + 1
    else:
        assert cc.train_X.size()[0] == 1

    # assert that the right elements have been added to the covariate observation
    for i in range(cc.train_X.size()[1]):
        assert cc.train_X[-1, i].item() == candidate_tensor[0, i].item()

    # assert that a new observation has been added for the response
    if train_Y is not None:
        assert cc.train_Y.size()[0] == train_Y.size()[0] + 1
    else:
        assert cc.train_Y.size()[1] == 1

    # assert that the right elements have been added to the response observation
    assert cc.train_Y[-1, 0].item() == kwarg_response[0,0].item() #resp_tensor[0, 0].item()

    ### check that acquisition function and model have been added

    # check that a model function has been assigned (should happen in all cases as part of tell)
    assert cc.model["model"] is not None

    # check that an acquisition function has been added (only if some data present in train_X, train_Y at first step)
    if train_X is not None:
        assert cc.acq_func["object"] is not None


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter, kwarg_covariates, kwarg_response, random_start",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, torch.tensor([[0.7]], dtype=torch.double), torch.tensor([[1.8]], dtype=torch.double), False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, torch.tensor([[0.7]], dtype=torch.double), torch.tensor([[1.8]], dtype=torch.double), False],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, torch.tensor([[0.7]], dtype=torch.double), torch.tensor([[1.8]], dtype=torch.double), True],  # case 1 with random start
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0, torch.tensor([[0.7]], dtype=torch.double), torch.tensor([[1.8]], dtype=torch.double), True],  # case 2 with random start
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[0.7]], dtype=torch.double), torch.tensor([[1.8]], dtype=torch.double), False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1.8, 1.2, 107]], dtype=torch.double), torch.tensor([[103]], dtype=torch.double), False],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1.8, 1.2, 107]], dtype=torch.double), torch.tensor([[0.8]], dtype=torch.double), False]
    ]
)
def test_CreativeProject_integration_ask_tell_one_loop_kwarg_covars_response_works(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, kwarg_covariates,
                                                                                   random_start, kwarg_response):
    """
    test that a single loop of ask/tell works when providing both covariates and response as kwarg to tell: creates a
    candidate, creates a model, stores covariates and response.
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=random_start)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter
    if covars_proposed_iter > 0:
        cc.num_initial_random_points = 0
        cc.random_sampling_method = None

    # run the ask method
    cc.ask()

    # run the tell method
    cc.tell(covars=kwarg_covariates, response=kwarg_response)


    ### check for tell (no reason to assert for ask)###

    # assert that a new observation has been added for covariates
    if train_X is not None:
        assert cc.train_X.size()[0] == train_X.size()[0] + 1
    else:
        assert cc.train_X.size()[0] == 1

    # assert that the right elements have been added to the covariate observation
    for i in range(cc.train_X.size()[1]):
        assert cc.train_X[-1, i].item() == kwarg_covariates[0, i].item()

    # assert that a new observation has been added for the response
    if train_Y is not None:
        assert cc.train_Y.size()[0] == train_Y.size()[0] + 1
    else:
        assert cc.train_Y.size()[1] == 1

    # assert that the right elements have been added to the response observation
    assert cc.train_Y[-1, 0].item() == kwarg_response[0,0].item()

    ### check that acquisition function and model have been added

    # check that a model function has been assigned (should happen in all cases as part of tell)
    assert cc.model["model"] is not None

    # check that an acquisition function has been added (only if some data present in train_X, train_Y at first step)
    if train_X is not None:
        assert cc.acq_func["object"] is not None


# test in single ask-tell loop for failures
@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter, kwarg_covariates, error_msg",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, torch.tensor([[1.8, 2.0]], dtype=torch.double), "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like 'tensor([1.], dtype=torch.float64)', but got 'tensor([[1.8000, 2.0000]], dtype=torch.float64)'"],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, [1, 'a'], "must be real number, not str"],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1.8, 2.2]], dtype=torch.double), "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like 'tensor([1.5000], dtype=torch.float64)', but got 'tensor([[1.8000, 2.2000]], dtype=torch.float64)'"],
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, ['b', 12.5], "too many dimensions 'str'"],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[0.8, 0.2, 103, 12]], dtype=torch.double), "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in 3 iterations. Was expecting something like 'tensor([  1.1355,  -3.1246, 105.4396], dtype=torch.float64)', but got 'tensor([[  0.8000,   0.2000, 103.0000,  12.0000]], dtype=torch.float64)'"],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([0.8, 0.2, 103], dtype=torch.double), "creative_project.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'. Was expecting torch tensor of size (1,<num_covariates>) but received one of size [3]"],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, [1, 2, 'a'], "must be real number, not str"]
    ]
)
def test_CreativeProject_integration_ask_tell_one_loop_kwarg_covars_fails(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, kwarg_covariates, error_msg,
                                                                          monkeypatch):
    """
    test that a single loop of ask/tell fails when providing covars as kwarg to tell. Monkeypatch
    "_read_response_manual_input" from ._observe.py to circumvent manual input via builtins.input and provides
    covariates via kwargs
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=False)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter
    if covars_proposed_iter > 0:
        cc.num_initial_random_points = 0
        cc.random_sampling_method = None

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)

    def mock_read_response_manual_input(additional_text):
        return resp_tensor
    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the ask method
    cc.ask()

    # run the tell method
    with pytest.raises(Exception) as e:
        cc.tell(covars=kwarg_covariates)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter, kwarg_response, error_msg",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, torch.tensor([[1.8, 2.2]], dtype=torch.double), "creative_project._observe._get_and_verify_response_input: incorrect number of variables provided. Was expecting input of size (1,1) but received torch.Size([1, 2])"],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, ['a'], "too many dimensions 'str'"],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0, [12, 'a'], "must be real number, not str"],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[1.8, 2.2]], dtype=torch.double), "creative_project._observe._get_and_verify_response_input: incorrect number of variables provided. Was expecting input of size (1,1) but received torch.Size([1, 2])"],
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, ['b', 12.5], "too many dimensions 'str'"],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, [0.8, 'b'], "must be real number, not str"],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([[0.8], [103]], dtype=torch.double), "creative_project.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'. Was expecting torch tensor of size (1,<num_covariates>) but received one of size [2, 1]"],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, torch.tensor([0.8, 103], dtype=torch.double), "creative_project.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'. Was expecting torch tensor of size (1,<num_covariates>) but received one of size [2]"],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1, [1, 'a'], "must be real number, not str"]
    ]
)
def test_CreativeProject_integration_ask_tell_one_loop_kwarg_response_fails(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, kwarg_response, error_msg,
                                                                          monkeypatch):
    """
    test that a single loop of ask/tell fails when providing covars as kwarg to tell. Monkeypatch
    "_read_response_manual_input" from ._observe.py to circumvent manual input via builtins.input and provides
    covariates via kwargs
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=False)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter
    if covars_proposed_iter > 0:
        cc.num_initial_random_points = 0
        cc.random_sampling_method = None

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)
    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor
    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # run the ask method
    cc.ask()

    # run the tell method
    with pytest.raises(Exception) as e:
        cc.tell(response=kwarg_response)
    assert str(e.value) == error_msg


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0],  # the case where no data is available (starts by training model)
    ]
)
def test_CreativeProject_integration_ask_tell_ask_works(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, monkeypatch):
    """
    test that an iteration of ask-tell-ask works (like "test_CreativeProject_tell_integration_ask_tell_one_loop_works"
    above. Specifically also test that this stores an acquisition function. Monkeypatch "_read_covars_manual_input"
    and "_read_response_manual_input" from ._observe.py to circumvent manual input via builtins.input
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=False)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)

    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor

    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)

    def mock_read_response_manual_input(additional_text):
        return resp_tensor

    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the ask method
    cc.ask()

    # run the tell method
    cc.tell()

    # run the ask method for a new data point
    cc.ask()

    # check acqusition function
    assert cc.acq_func["object"] is not None

    # check that a model function has been assigned (should happen in all cases as part of tell)
    assert cc.model["model"] is not None

    ### assert for ask ###

    # check that TWO entries has been added to cls.proposed_X
    assert cc.proposed_X.size()[0] == 2

    # assert that the number of covars in returned "proposed_X" matches the number from "covars"
    assert cc.proposed_X.size()[1] == len(covars)

    # check that the counter 'covars_proposed_iter' is updated TWICE
    assert cc.model["covars_proposed_iter"] == covars_proposed_iter + 2

    ### check for tell ###

    # assert that ONE observation has been added for covariates
    assert cc.train_X.size()[0] == 1

    # assert that the right elements have been added to the covariate observation
    for i in range(cc.train_X.size()[1]):
        assert cc.train_X[-1, i].item() == candidate_tensor[0, i].item()

    # assert that ONE observation has been added for the response
    assert cc.train_Y.size()[1] == 1

    # assert that the right elements have been added to the response observation
    assert cc.train_Y[-1, 0].item() == resp_tensor[0, 0].item()


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1]
    ]
)
def test_CreativeProject_integration_ask_ask_tell_overwrite_candidate_works(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, monkeypatch):
    """
    test that the first proposed new candidate datapoint is ignored if ask is run twice (without any tell). Test that
    everything works downstream: creates a candidate, creates a model, stores covariates and response.
    Monkeypatch "_read_covars_manual_input" and "_read_response_manual_input" from ._observe.py to circumvent manual
    input via builtins.input
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=False)

    # set attributes on class (to simulate previous iterations of ask/tell functionality). That is, set attributes set
    # both by _Initializers__initialize_training_data and by _Initializers__initialize_random_start
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter
    cc.num_initial_random_points = 0
    cc.random_sampling_method = None


    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)

    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor

    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)

    def mock_read_response_manual_input(additional_text):
        return resp_tensor

    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the ask method
    cc.ask()

    # run the ask method AGAIN
    cc.ask()

    # run the tell method
    cc.tell()

    ### assert for ask ###

    # check that an entry has been added to cls.proposed_X
    assert cc.proposed_X.size()[0] == train_X.size()[0] + 1
    assert cc.proposed_X[-1].size()[0] == train_X.size()[1]  # check that the new candidate has the right number of entries

    # assert that the number of covars in returned "proposed_X" matches the number from "covars"
    assert cc.proposed_X.size()[1] == len(covars)

    # check that the counter 'covars_proposed_iter' is updated ONLY ONCE
    assert cc.model["covars_proposed_iter"] == covars_proposed_iter + 1

    ### check for tell ###

    # assert that ONE new observation has been added for covariates
    assert cc.train_X.size()[0] == train_X.size()[0] + 1

    # assert that the right elements have been added to the covariate observation
    for i in range(cc.train_X.size()[1]):
        assert cc.train_X[-1, i].item() == candidate_tensor[0, i].item()

    # assert that ONE new observation has been added for the response
    assert cc.train_Y.size()[0] == train_Y.size()[0] + 1

    # assert that the right elements have been added to the response observation
    assert cc.train_Y[-1, 0].item() == resp_tensor[0, 0].item()

# test with repeat "tell" that train_X, train_Y last row is only added once
@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1]
    ]
)
def test_CreativeProject_integration_ask_tell_tell_overwrite_covar_resp_works(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, monkeypatch):
    """
    test that the first reported datapoint for covars and response (last entries in train_X, train_Y) are overwritten if
    tell is run twice (without two iterations of ask). Only the last datapoint entry should remain. Test that
    everything works downstream: creates a candidate, creates a model, stores covariates and response.
    Monkeypatch "_read_covars_manual_input" and "_read_response_manual_input" from ._observe.py to circumvent manual
    input via builtins.input
    """

    # initialize the class
    cc = CreativeProject(covars=covars, model=model_type, random_start=False)

    # ISSUE IS THAT I AM CIRCUMVENTING THE INITIALIZATION OF RANDOM POINTS WHICH REQUIRES THAT TRAIN_X, TRAIN_Y
    # INITIALIZATION HAS FINISHED. I NEED TO SET RANDOM INITIALIZATION PARAMTERES MANUALLY BELOW
    #
    # ALSO MAKE SURE I DO AT LEAST ONE TEST WHERE I DO THE FULL E2E TEST (ALLOWING FOR AUTOMATICALLY CREATING RANDOM
    # INITIALIZATION)

    # set attributes on class (to simulate previous iterations of ask/tell functionality). That is, set attributes set
    # both by _Initializers__initialize_training_data and by _Initializers__initialize_random_start
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter
    cc.num_initial_random_points = 0
    cc.random_sampling_method = None

    # define decorator to add 1.0 to all entries in monkeypatched returned data. This to be able to tell that the last
    # entry (from second "tell") is different than the first, and know that it has been overwritten
    def add_one(func):
        @functools.wraps(func)
        def wrapper_add_one(*args, **kwargs):
            wrapper_add_one.num_calls += 1
            output = func(*args, **kwargs)
            return output + wrapper_add_one.num_calls
        wrapper_add_one.num_calls = 0
        return wrapper_add_one

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)

    @add_one
    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor
    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)

    @add_one
    def mock_read_response_manual_input(additional_text):
        return resp_tensor
    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the ask method
    cc.ask()

    # run the tell method
    cc.tell()

    # run the tell method AGAIN
    cc.tell()

    ### assert for ask ###

    # check that an entry has been added to cls.proposed_X
    assert cc.proposed_X.size()[0] == train_X.size()[0] + 1
    assert cc.proposed_X[-1].size()[0] == train_X.size()[1]  # check that the new candidate has the right number of entries

    # assert that the number of covars in returned "proposed_X" matches the number from "covars"
    assert cc.proposed_X.size()[1] == len(covars)

    # check that the counter 'covars_proposed_iter' is updated
    assert cc.model["covars_proposed_iter"] == covars_proposed_iter + 1

    ### check for tell ###

    # assert that ONLY ONE new observation has been added for covariates
    assert cc.train_X.size()[0] == train_X.size()[0] + 1

    # assert that the right elements have been added to the covariate observation (should be candidate_tensor with
    # "add_one" applied twice, i.e. adding 2.0 to each entry)
    for i in range(cc.train_X.size()[1]):
        assert cc.train_X[-1, i].item() == candidate_tensor[0, i].item() + 2.0

    # assert that ONLY ONE new observation has been added for the response
    assert cc.train_Y.size()[0] == train_Y.size()[0] + 1

    # assert that the right elements have been added to the response observation (should be resp_tensor with "add_one"
    # applied twice, i.e. adding 2.0 to each entry)
    assert cc.train_Y[-1, 0].item() == resp_tensor[0, 0].item() + 2.0


# test that model is updated (overwritten)
@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0],  # the case where no data is available (starts by training model)
    ]
)
def test_CreativeProject_integration_ask_tell_ask_works(covars, model_type, train_X, train_Y,
                                                                  covars_proposed_iter, covars_sampled_iter,
                                                                  response_sampled_iter, monkeypatch):
    """
    test that both surrogate model and acquisition functions are added and updated following two rounds of ask-tell.
    Monkeypatch "_read_covars_manual_input" and "_read_response_manual_input" from ._observe.py to circumvent manual
    input via builtins.input. This automatically tests the new functionality of random start by starting from no data
    (train_X, train_Y)
    """

    # initialize the class
    # random_start = True is default, so this tests random start
    cc = CreativeProject(covars=covars, model=model_type)

    # set attributes on class (to simulate previous iterations of ask/tell functionality)
    cc.train_X = train_X
    cc.proposed_X = train_X
    cc.train_Y = train_Y
    cc.model["covars_proposed_iter"] = covars_proposed_iter
    cc.model["covars_sampled_iter"] = covars_sampled_iter
    cc.model["response_sampled_iter"] = response_sampled_iter

    # define decorator to add 1.0 to all entries in monkeypatched returned data. This to be able to tell that the last
    # entry (from second "tell") is different than the first, and know that it has been overwritten
    def add_one(func):
        @functools.wraps(func)
        def wrapper_add_one(*args, **kwargs):
            wrapper_add_one.num_calls += 1
            output = func(*args, **kwargs)
            return output + wrapper_add_one.num_calls

        wrapper_add_one.num_calls = 0
        return wrapper_add_one

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)

    @add_one
    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor

    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)

    @add_one
    def mock_read_response_manual_input(additional_text):
        return resp_tensor

    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # run the ask method
    cc.ask()

    # run the tell method
    cc.tell()

    # grab the model state
    surrogate_model1 = cc.model["model"]

    # run the ask method AGAIN
    cc.ask()

    # grab the acquisition function
    acq_func1 = cc.acq_func["object"]

    # run the tell method AGAIN
    cc.tell()

    # grab the model state
    surrogate_model2 = cc.model["model"]

    # run the ask method a THIRD TIME
    cc.ask()

    # grab the acquisition function
    acq_func2 = cc.acq_func["object"]

    # assert that both model and acquisition functions exist
    assert cc.model["model"] is not None
    assert cc.acq_func["object"] is not None

    # assert that surrogate model has updated
    assert surrogate_model1 != surrogate_model2

    # assert that acquisition function has updated
    assert acq_func1 != acq_func2


@pytest.mark.parametrize(
    "train_X, train_Y, random_sampling_method",
    [
        [None, None, "random"],
        [None, None, "latin_hcs"],
        [torch.tensor([[1.1, 2.1, 23.7]], dtype=torch.double), torch.tensor([[10.7]], dtype=torch.double), "random"],
        [torch.tensor([[1.1, 2.1, 23.7],[1.9, 1.8, 18.2]], dtype=torch.double), torch.tensor([[10.7], [13.2]], dtype=torch.double), "random"],
        [torch.tensor([[1.1, 2.1, 23.7],[1.9, 1.8, 18.2]], dtype=torch.double), torch.tensor([[10.7], [13.2]], dtype=torch.double), "latin_hcs"],
    ]
)
def test_CreativeProject_integration_ask_tell_ask_tell_randon_start_works(train_X, train_Y, random_sampling_method,
                                                                          monkeypatch):
    """
    test that ask-tell dynamics works with random start, with and without train_X, train_Y data being provided.
    Monkeypatching user input
    """

    covars = [(1, 0, 2), (1.5, -1, 3), (22.0, 15, 27)]
    num_initial_random = 1

    # initialize the class
    cc = CreativeProject(covars=covars, train_X=train_X, train_Y=train_Y, random_start=True,
                         random_sampling_method=random_sampling_method, num_initial_random=num_initial_random)

    # define decorator to add 1.0 to all entries in monkeypatched returned data. This to be able to tell that the last
    # entry (from second "tell") is different than the first, and know that it has been overwritten
    def add_one(func):
        @functools.wraps(func)
        def wrapper_add_one(*args, **kwargs):
            wrapper_add_one.num_calls += 1
            output = func(*args, **kwargs)
            return output + wrapper_add_one.num_calls

        wrapper_add_one.num_calls = 0
        return wrapper_add_one

    # monkeypatch "_read_covars_manual_input"
    candidate_tensor = torch.tensor([[tmp[0] for tmp in covars]], dtype=torch.double)

    @add_one
    def mock_read_covars_manual_input(additional_text):
        return candidate_tensor

    monkeypatch.setattr(cc, "_read_covars_manual_input", mock_read_covars_manual_input)

    # monkeypatch "_read_response_manual_input"
    resp_tensor = torch.tensor([[12]], dtype=torch.double)

    @add_one
    def mock_read_response_manual_input(additional_text):
        return resp_tensor

    monkeypatch.setattr(cc, "_read_response_manual_input", mock_read_response_manual_input)

    # check the number of iterations we're starting from
    curr_iter = 0
    if train_X is not None:  # enough to look at train_X since validator ensures train_X, train_Y have same number of rows
        curr_iter = train_X.size()[0]

    assert cc.model["covars_proposed_iter"] == curr_iter
    assert cc.model["covars_sampled_iter"] == curr_iter
    assert cc.model["response_sampled_iter"] == curr_iter

    # run the ask method
    cc.ask()

    # run the tell method
    cc.tell()

    # assert that counters have increased by 1
    curr_iter += 1
    assert cc.model["covars_proposed_iter"] == curr_iter
    assert cc.model["covars_sampled_iter"] == curr_iter
    assert cc.model["response_sampled_iter"] == curr_iter

    # run the ask method AGAIN
    cc.ask()

    # run the tell method AGAIN
    cc.tell()

    # assert that counters have increaed by 1 yet again (this time switching from random to bayesian)
    curr_iter += 1
    assert cc.model["covars_proposed_iter"] == curr_iter
    assert cc.model["covars_sampled_iter"] == curr_iter
    assert cc.model["response_sampled_iter"] == curr_iter

