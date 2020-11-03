import pytest
import torch


def test_update_max_response_value_unit(custom_models_simple_training_data_4elements, tmp_best_response_class):
    """
    test that _update_max_response_value works for univariate data
    """

    # data -- max at 4th element
    max_index = 3
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # test class
    cls = tmp_best_response_class

    # set some attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y

    # test that it initializes storage of best results when none present
    cls._update_max_response_value()

    assert cls.covars_best_response_value[0].item() == train_X[max_index].item()
    assert cls.best_response_value[0].item() == train_Y[max_index].item()


def test_update_max_response_value_fail_functional(tmp_best_response_class):
    """
    test that _update_max_response_value fails for univariate data
    """

    # data
    train_X = None
    train_Y = None

    # test class
    cls = tmp_best_response_class

    # set some attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y

    # test that it initializes storage of best results when none present
    with pytest.raises(Exception) as e:
        cls._update_max_response_value()
    assert str(e.value) == "creative_project._best_response._update_max_response_value.py: Missing or unable to process " \
                           "one of following attributes: self.train_X, self.train_Y"


@pytest.mark.parametrize("initial_best_covars, initial_best_response",
                         [
                             [None, None],
                             [torch.tensor([[1.0, 2.0, 1.0]], dtype=torch.double),
                              torch.tensor([[0.2]], dtype=torch.double)]
                         ]
                         )
def test_update_max_response_value_multivariate_functional(training_data_covar_complex,
                                                     tmp_best_response_class, initial_best_covars,
                                                     initial_best_response):
    """
    test that _update_max_response_value works for multivariate data. Test both if no data present and some data already
    present (training data, with best solution captured)
    """

    # data -- max at 2nd element
    max_index = 1
    covars = training_data_covar_complex[0]
    train_X = training_data_covar_complex[1]
    train_Y = training_data_covar_complex[2]

    # test class
    cls = tmp_best_response_class

    # set some attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y

    cls.covars_best_response_value = initial_best_covars
    cls.best_response_value = initial_best_response

    # test that it initializes storage of best results when none present
    cls._update_max_response_value()

    # for first pass of the test, there will only be only one row in attributes covars_best_response_value,
    # best_response_value; for second pass there will be 2. In both cases, the LAST ROW corresponds to assessing the
    # new field just added
    for it in range(train_X.shape[1]):
        assert cls.covars_best_response_value[-1, it].item() == train_X[max_index, it].item()
    assert cls.best_response_value[-1].item() == train_Y[max_index].item()
