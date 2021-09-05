import pandas as pd
import pytest
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model


def test_update_max_response_value_unit(custom_models_simple_training_data_4elements, tmp_best_response_class,
                                        custom_models_simple_training_data_4elements_covar_details):
    """
    test that _update_max_response_value works for univariate data
    """

    # data -- max at 4th element
    max_index = 3
    train_X = custom_models_simple_training_data_4elements[0]
    train_Y = custom_models_simple_training_data_4elements[1]
    covar_details = custom_models_simple_training_data_4elements_covar_details[0]
    covar_mapped_names = custom_models_simple_training_data_4elements_covar_details[1]

    # test class
    cls = tmp_best_response_class

    # set some attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y
    cls.covar_details = covar_details
    cls.covar_mapped_names = covar_mapped_names

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
    assert str(e.value) == "greattunes._best_response._update_max_response_value.py: Missing or unable to process " \
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
    covar_details = training_data_covar_complex[3]
    covar_mapped_names = training_data_covar_complex[4]

    # test class
    cls = tmp_best_response_class

    # set some attributes
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y
    cls.covar_details = covar_details
    cls.covar_mapped_names = covar_mapped_names
    cls.x_data = pd.DataFrame(columns=covar_mapped_names)
    cls.y_data = pd.DataFrame(columns=["Response"])

    cls.covars_best_response_value = initial_best_covars
    cls.best_response_value = initial_best_response
    if initial_best_covars is None:
        cls.covars_best_response = cls.x_data
        cls.best_response = cls.y_data

    else:
        cls.covars_best_response = pd.DataFrame({"covar0": [initial_best_covars[0,0].item()], "covar1": [initial_best_covars[0,1].item()], "covar2": [initial_best_covars[0,2].item()]})
        cls.best_response = pd.DataFrame({"Response": [initial_best_response[0,0].item()]})

    # test that it initializes storage of best results when none present
    cls._update_max_response_value()

    # for first pass of the test, there will only be only one row in attributes covars_best_response_value,
    # best_response_value; for second pass there will be 2. In both cases, the LAST ROW corresponds to assessing the
    # new field just added
    for it in range(train_X.shape[1]):
        assert cls.covars_best_response_value[-1, it].item() == train_X[max_index, it].item()
    assert cls.best_response_value[-1].item() == train_Y[max_index].item()


def test_best_predicted_integration(training_data_covar_complex, tmp_best_response_class, capsys):
    """
    test that best_predicted method runs produces the right prompt output
    """

    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data -- max at 2nd element
    max_index = 1
    covars = training_data_covar_complex[0]
    train_X = training_data_covar_complex[1]
    train_Y = training_data_covar_complex[2]
    covar_details = training_data_covar_complex[3]
    covar_mapped_names = training_data_covar_complex[4]

    initial_guesses = torch.tensor([[x[0] for x in covars]], dtype=dtype, device=device)
    covar_bounds = torch.tensor([[x[1] for x in covars], [x[2] for x in covars]], dtype=dtype, device=device)

    # test class
    cls = tmp_best_response_class

    # add some attributes to tmp
    cls.dtype = torch.double
    cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # add attributes to class
    cls.train_X = train_X
    cls.proposed_X = train_X
    cls.train_Y = train_Y
    cls.covar_details = covar_details
    cls.covar_mapped_names = covar_mapped_names
    cls.initial_guesses = initial_guesses
    cls.covar_bounds = covar_bounds

    # define and train model
    mymodel = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    ll = ExactMarginalLogLikelihood(mymodel.likelihood, mymodel)
    fit_gpytorch_model(ll)

    cls.model = {"model": mymodel}

    # run the method
    cls.best_predicted()

    captured = capsys.readouterr()

    # construct the expected output
    outtext = "Best predicted response value Y (mean model): max_Y = 4.87768e+00\n" \
              "Corresponding covariate values resulting in max_Y:\n\t    name\n\t-2.50338\n\n" \
              "Best predicted response value Y (lower confidence region): max_Y = 2.87374e+00\n" \
              "Corresponding covariate values resulting in max_Y:\n\t     name\n\t-2.500754\n"

    assert captured.out == outtext
