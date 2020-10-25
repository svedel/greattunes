import pytest
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
