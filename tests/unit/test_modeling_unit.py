import torch
import pandas as pd
import pytest
import types
from greattunes._modeling import _set_GP_model, _mapped_noise_from_model


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
    assert str(e.value) == "greattunes._modeling._set_GP_model: unknown 'model_type' " \
                           "(" + cls.model["model_type"] + ") provided. Must be in following list " \
                            "['FixedNoiseGP', 'HeteroskedasticSingleTaskGP', 'SingleTaskGP', 'SimpleCustomMaternGP']"


@pytest.mark.parametrize(
    "y_data, train_Yvar",
    [
        [pd.DataFrame({"Response": [1.0, 2.0, 3.0]}), torch.tensor(0.25, dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))],  # tensor of dimension 0
        [pd.DataFrame({"Response": [1.0, 2.0, 3.0]}), torch.tensor([0.25], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))],  # tensor of dimension 1
        [pd.DataFrame({"Response": [1.0, 2.0, 3.0]}), 0.33],  # float
        [pd.DataFrame({"Response": [1.0, 2.0, 3.0]}), 1],  # int
        [pd.DataFrame({"Response": [1.0, 2.0, 3.0]}), lambda y_data: y_data + 0.25],  # a function, here using a lambda function
    ]
)
def test_modeling__mapped_noise_from_model_works(y_data, train_Yvar):
    """
    tests that _mapped_noise_from_model method can correctly run if train_Yvar is a tensor of dimension 0 and 1, a
    floats, and that it can run if train_Yvar is a function
    """

    # define temporary class
    class Tmp:
        def __init__(self, y_data, train_Yvar):
            self.y_data = y_data
            self.train_Yvar = train_Yvar
            self.train_Y = torch.from_numpy(y_data.values)

            self.dtype = torch.double
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from greattunes._modeling import _mapped_noise_from_model

    cls = Tmp(y_data=y_data, train_Yvar=train_Yvar)

    # run the method
    train_Yvar_mapped = cls._mapped_noise_from_model()

    # assert size
    assert y_data.shape[0] == train_Yvar_mapped.size()[0]

    # assert output type
    assert isinstance(train_Yvar_mapped, torch.DoubleTensor)

    # special case for cases which are not functions, to assert level of noise
    if not isinstance(cls.train_Yvar, types.FunctionType):
        if isinstance(cls.train_Yvar, torch.DoubleTensor):
            if len(list(cls.train_Yvar.size())) == 0:
                assert train_Yvar_mapped[0,0].item() == train_Yvar.item()
            elif len(list(cls.train_Yvar.size())) == 1:
                assert train_Yvar_mapped[0,0].item() == train_Yvar[0].item()
        elif isinstance(cls.train_Yvar, float) or isinstance(cls.train_Yvar, int):
            assert train_Yvar_mapped[0,0].item() == float(train_Yvar)


@pytest.mark.parametrize(
    "y_data, train_Yvar",
    [
        [pd.DataFrame({"Response": [1.0, 2.0, 3.0]}), torch.tensor([[0.25]], dtype=torch.double, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))],  # tensor of dimension 2
        [pd.DataFrame({"Response": [1.0, 2.0, 3.0]}), "hello"],  # bool
        [pd.DataFrame({"Response": [1.0, 2.0, 3.0]}), None],  # None
    ]
)
def test_modeling__mapped_noise_from_model_fails(y_data, train_Yvar):
    """
    test that the method throws an error if if train_Yvar is a tensor of the wrong dimension, or if
    it is not a float or int
    """

    # define temporary class
    class Tmp:
        def __init__(self, y_data, train_Yvar):
            self.y_data = y_data
            self.train_Yvar = train_Yvar
            self.train_Y = torch.from_numpy(y_data.values)

            self.dtype = torch.double
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model = {"model_type": "SingleTaskGP"}

        from greattunes._modeling import _mapped_noise_from_model

    cls = Tmp(y_data=y_data, train_Yvar=train_Yvar)

    # run the method
    if isinstance(cls.train_Yvar, torch.DoubleTensor):  # special case if tensor is provided
        with pytest.raises(Exception) as e:
            train_Yvar_mapped = cls._mapped_noise_from_model()
        assert str(e.value) == "greattunes.greattunes._observe._mapped_noise_from_model: tensor provided for 'train_Yvar' has unacceptable dimensions. Only tensors of dimension 0 or 1 are accepted, provided tensor has dimension " + str(len(list(train_Yvar.size()))) + "."
    else:
        with pytest.raises(Exception) as e:
            train_Yvar_mapped = cls._mapped_noise_from_model()
        assert str(e.value) == "greattunes.greattunes._observe._mapped_noise_from_model: provided object for 'train_Yvar' is not acceptable. It must be either (i) a tensor of dimension 0 or 1, (ii) a float or int, or (iii) a function which operates on self.y_data. Provided object is of type " + str(type(train_Yvar)) + "."