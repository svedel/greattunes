import pytest
from creative_project._modeling import _set_GP_model


@pytest.mark.parametrize("nu", [1.5, None, 2])
def test_modeling__set_GP_model_fail_unit(training_data_covar_complex, tmp_modeling_class, nu):
    """
    test that setting model fails if unsupported model_type name provided. test across a few different values for nu
    """

    covars = training_data_covar_complex[0]
    train_X = training_data_covar_complex[1]
    train_Y = training_data_covar_complex[2]

    cls = tmp_modeling_class

    # set a few key attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y

    cls.model["model_type"] = "not_supported"
    cls.model["response_sampled_iter"] = cls.train_X.shape[0]

    # run the method
    with pytest.raises(Exception) as e:
        output_text = cls._set_GP_model(nu=nu)
    assert str(e.value) == "creative_project._modeling._set_GP_model: unknown 'model_type' " \
                           "(" + cls.model["model_type"] + ") provided. Must be in following list " \
                                                            "['Custom', 'SingleTaskGP']"
