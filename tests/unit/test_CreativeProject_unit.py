import pytest
import torch
from creative_project import CreativeProject


@pytest.mark.parametrize("nu", [None, 2.5])
def test_CreativeProject__init__covars_notrainingdata_works(covars_for_custom_models_simple_training_data_4elements, nu, monkeypatch):
    """
    test CreativeProject class initialization under conditions where it should work (no data provided). Kwarg nu is
    tested for None and numerical value
    """

    # data
    covars = covars_for_custom_models_simple_training_data_4elements

    # monkeypatch inherited private method _Initializers__initialize_from_covars
    def mock__initialize_from_covars(self, covars):
        guesses = [[g[0] for g in covars]]
        lb = [g[1] for g in covars]
        ub = [g[2] for g in covars]
        return torch.tensor(guesses, dtype=torch.double), torch.tensor([lb, ub], dtype=torch.double)
    monkeypatch.setattr(
        CreativeProject, "_Initializers__initialize_from_covars", mock__initialize_from_covars
    )

    # set up class
    cls = CreativeProject(covars=covars, nu=nu)

    # assert class settings
    assert cls.device == torch.device(type='cpu')
    assert isinstance(cls.dtype, torch.dtype)

    # assert that attributes set/not set for initialization
    assert cls.sampling["method"] == "manual"

    # assert initialization
    assert cls.initial_guess[0].item() == covars[0][0]
    assert cls.covar_bounds[0].item() == covars[0][1]
    assert cls.covar_bounds[1].item() == covars[0][2]

    # assert a few attributes
    assert cls.model["covars_sampled_iter"] == 0
    assert cls.model["model_type"] == "SingleTaskGP"
    assert cls.model["model"] is None

    # assert no data
    assert cls.proposed_X is None
    assert cls.train_X is None
    assert cls.train_Y is None

    # assert best response
    assert cls.covars_best_response_value is None
    assert cls.best_response_value is None

    # assert that kwargs nu is set
    assert cls.nu == nu


def test_CreativeProject__init__covars_trainingdata_works(covars_for_custom_models_simple_training_data_4elements,
                                                          custom_models_simple_training_data_4elements, monkeypatch):
    """
    tests class initialization with training data provided 
    """
    
    covars = covars_for_custom_models_simple_training_data_4elements
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # monkeypatch inherited private method _Initializers__initialize_from_covars
    def mock__initialize_from_covars(self, covars):
        guesses = [[g[0] for g in covars]]
        lb = [g[1] for g in covars]
        ub = [g[2] for g in covars]
        return torch.tensor(guesses, dtype=torch.double), torch.tensor([lb, ub], dtype=torch.double)
    monkeypatch.setattr(
        CreativeProject, "_Initializers__initialize_from_covars", mock__initialize_from_covars
    )

    # monkeypatch training data initialization
    def mock__initialize_training_data(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        return True
    monkeypatch.setattr(
        CreativeProject, "_Initializers__initialize_training_data", mock__initialize_training_data
    )

    # monkeypatch best response initialization
    tmp_val = 1.2
    def mock__initialize_best_response(self):
        self.covars_best_response_value = tmp_val
        self.best_response_value = tmp_val
        return True
    monkeypatch.setattr(
        CreativeProject, "_Initializers__initialize_best_response", mock__initialize_best_response
    )

    # initialize class
    cls = CreativeProject(covars=covars, train_X=train_X, train_Y=train_Y)

    # assert training data set
    assert cls.train_X is not None
    assert cls.train_X[0].item() == train_X[0].item()
    assert cls.train_Y is not None
    assert cls.train_Y[1].item() == train_Y[1].item()

    # assert best response initialized
    assert cls.covars_best_response_value == tmp_val
    assert cls.best_response_value == tmp_val

def test_CreativeProject__init__covars_trainingdata_none(covars_for_custom_models_simple_training_data_4elements,
                                                          monkeypatch):
    """
    tests class initialization with training data provided
    """

    covars = covars_for_custom_models_simple_training_data_4elements
    train_X = None
    train_Y = None

    # monkeypatch inherited private method _Initializers__initialize_from_covars
    def mock__initialize_from_covars(self, covars):
        guesses = [[g[0] for g in covars]]
        lb = [g[1] for g in covars]
        ub = [g[2] for g in covars]
        return torch.tensor(guesses, dtype=torch.double), torch.tensor([lb, ub], dtype=torch.double)
    monkeypatch.setattr(
        CreativeProject, "_Initializers__initialize_from_covars", mock__initialize_from_covars
    )

    # monkeypatch training data initialization
    def mock__initialize_training_data(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        return True
    monkeypatch.setattr(
        CreativeProject, "_Initializers__initialize_training_data", mock__initialize_training_data
    )

    # monkeypatch best response initialization
    tmp_val = None
    def mock__initialize_best_response(self):
        self.covars_best_response_value = tmp_val
        self.best_response_value = tmp_val
        return True
    monkeypatch.setattr(
        CreativeProject, "_Initializers__initialize_best_response", mock__initialize_best_response
    )

    # initialize class
    cls = CreativeProject(covars=covars, train_X=train_X, train_Y=train_Y)

    # assert training data
    assert cls.train_X is None
    assert cls.train_Y is None

    # assert best response initialized
    assert cls.covars_best_response_value is tmp_val
    assert cls.best_response_value is tmp_val