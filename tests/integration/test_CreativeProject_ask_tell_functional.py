import pytest
import torch
from creative_project import CreativeProject


@pytest.mark.parametrize(
    "covars, model_type, train_X, train_Y, covars_proposed_iter, covars_sampled_iter, response_sampled_iter",
    [
        [[(1, 0.5, 1.5)], "SingleTaskGP", None, None, 0, 0, 0],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "Custom", None, None, 0, 0, 0],  # the case where no data is available (starts by training model)
        [[(1, 0.5, 1.5)], "SingleTaskGP", torch.tensor([[0.8]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "SingleTaskGP", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1],
        [[(1, 0.5, 1.5), (-3, -4, 1.1), (100, 98.0, 106.7)], "Custom", torch.tensor([[0.8, 0.2, 102]], dtype=torch.double), torch.tensor([[22]], dtype=torch.double), 1, 1, 1]
    ]
)
def test_CreativeProject_ask_integration_test_works(covars, model_type, train_X, train_Y, covars_proposed_iter,
                                                    covars_sampled_iter, response_sampled_iter):
    """
    test the positive cases for CreativeProject.ask method.
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


# add test that integrates one round of ask and tell into a single test