import pytest
from creative_project._modeling import _set_GP_model


@pytest.mark.parametrize(
    "model_type, nu",
    [
        ["SingleTaskGP", None],
        ["Custom", 1.5]
    ]
)
def test_modeling__set_GP_model_functional(training_data_covar_complex, tmp_modeling_class, model_type, nu):
    """
    test that setting model works
    """

    covars = training_data_covar_complex[0]
    train_X = training_data_covar_complex[1]
    train_Y = training_data_covar_complex[2]

    cls = tmp_modeling_class

    # set a few key attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y

    cls.model["model_type"] = model_type
    cls.model["response_sampled_iter"] = cls.train_X.shape[0]

    # run the method
    output_text = cls._set_GP_model(nu=nu)

    assert output_text == "Successfully trained GP model"
    assert cls.model["likelihood"] is not None
    assert cls.model["loglikelihood"] is not None