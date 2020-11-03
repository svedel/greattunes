from creative_project._initializers import Initializers
from creative_project._best_response import _find_max_response_value
import torch


def test_Initializers__initialize_best_response_functional(custom_models_simple_training_data_4elements,
                                                           tmp_Initializers_with_find_max_response_value_class):
    """
    test initialization of best response data structures based on input data
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]

    # create test version of Initializers to endow it with the property from _find_max_response_value, which is
    # otherwise defined as a static method in ._best_response
    #class TmpClass(Initializers):
    #    from creative_project._best_response import _find_max_response_value
    #cls = TmpClass()
    cls = tmp_Initializers_with_find_max_response_value_class

    # initialize class and register required attributes
    #cls = Initializers()
    #
    # add validation method, only needed for test purposes
    #cls._find_max_response_value = _find_max_response_value

    cls.train_X = train_X
    cls.train_Y = train_Y

    # define required attributes for test to pass (IRL set in CreativeProject which is a child of Initializers)
    cls.dtype = torch.double
    cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run the function
    cls._Initializers__initialize_best_response()

    # assert that the function has been applied
    assert isinstance(cls.covars_best_response_value, torch.Tensor)
    assert isinstance(cls.best_response_value, torch.Tensor)

    # check size
    assert cls.covars_best_response_value.shape[0] == train_X.shape[0]
    assert cls.best_response_value.shape[0] == train_Y.shape[0]

    # check values, compare to mock_find_max_response_value
    # below two "test"-tensors contain best values corresponding to data from
    # conftest.py's "custom_models_simple_training_data_4elements"
    test_max_covars_test = torch.tensor([[-1.0], [-1.0], [-0.5], [1.0]], dtype=torch.double)
    test_max_response_test = torch.tensor([[0.2], [0.2], [0.5], [2.0]], dtype=torch.double)
    for it in range(train_X.shape[0]):
        assert cls.covars_best_response_value[it].item() == test_max_covars_test[it].item()
        assert cls.best_response_value[it].item() == test_max_response_test[it].item()


def test_Initializers__initialize_training_data_functional(custom_models_simple_training_data_4elements,
                                                           covars_for_custom_models_simple_training_data_4elements):
    """
    test private method initialize_training_data which depends on private method from parent class Validators. Test
    only functional and non-functional via provided initial data train_X, train_Y
    """

    # data
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]
    covars = covars_for_custom_models_simple_training_data_4elements

    ### First test: it passes (use train_X, train_Y)
    # initialize class and register required attributes
    cls = Initializers()
    cls.model = {  # set required attributes
        "covars_sampled_iter": 0,
        "response_sampled_iter": 0
    }
    cls.initial_guess = torch.tensor([[g[0] for g in covars]], dtype=torch.double)

    # run the method
    cls._Initializers__initialize_training_data(train_X=train_X, train_Y=train_Y)

    # assert that the data has been validated and stored in right places
    assert cls.start_from_guess == False
    for it in range(train_X.shape[0]):
        assert cls.train_X[it].item() == train_X[it].item()
        assert cls.train_Y[it].item() == train_Y[it].item()
        assert cls.proposed_X[it].item() == train_X[it].item()

    ### First test: it passes (use train_X, train_Y)
    # initialize class and register required attributes
    cls = Initializers()
    cls.model = {  # set required attributes
        "covars_sampled_iter": 0,
        "response_sampled_iter": 0
    }

    # run the method
    cls._Initializers__initialize_training_data(train_X=None, train_Y=None)

    # assert that nothing has run
    assert cls.start_from_guess == True
    assert cls.train_X == None
    assert cls.train_Y == None
    assert cls.proposed_X == None
