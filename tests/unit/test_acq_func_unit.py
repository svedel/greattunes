import pytest
import botorch
from creative_project._acq_func import AcqFunction

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
        "type": "EI", # define the type of acquisition function
        "object": None
    }

    # set attributes needed for test: train_Y to not None, model to None
    cls.train_Y = train_Y
    cls.model = {"model": None}

    with pytest.raises(Exception) as e:
        assert cls.set_acq_func()
    assert str(e.value) == "kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: no surrogate model set (self.model['model'] is None)"

    # set attributes needed for test: train_Y to not None, model to "something" (something that doesn't trigger exception)
    cls.train_Y = None
    cls.model = {"model": "something"}

    with pytest.raises(Exception) as e:
        assert cls.set_acq_func()
    assert str(e.value) == "kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: no training data provided (self.train_Y is None"

    # set attributes needed for test: train_Y to not None, model to None. Model exception should fire first
    cls.train_Y = None
    cls.model = {"model": None}

    with pytest.raises(Exception) as e:
        assert cls.set_acq_func()
    assert str(e.value) == "kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: no surrogate model set (self.model['model'] is None)"


def test_acq_func_set_acq_func_works(ref_model_and_training_data):
    """
    test that set_acq_func works with model and training data are provided correctly
    """

    # load data and model
    train_X = ref_model_and_training_data[0]
    train_Y = ref_model_and_training_data[1]
    model_obj = ref_model_and_training_data[2]
    lh = ref_model_and_training_data[3]
    ll = ref_model_and_training_data[4]

    # the acq func
    cls = AcqFunction()
    cls.acq_func = {
        "type": "EI",  # define the type of acquisition function
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