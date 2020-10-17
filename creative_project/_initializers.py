import torch
import creative_project._max_response
from creative_project._validators import Validators


class Initializers(Validators):
    """
    initializer functions used to initialize data structures etc. These functions are to remain private, i.e.
    non-accessible to the user.

    These methods are kept in a separate class to maintain that they be private in the main
    class (that class being defined in __init__.py).

    For simplicity is developed as child of Validators class (from creative_project._validators), making all validator
    methods available inside Initializers
    """

    def __initialize_from_covars(self, covars):
        """
        initialize covars for modeing. Currently only extract starting guess for each covariate, ASSUME each variable
        is continuous.
        TODO:
            - ALSO CONSIDER CHANGING LENGTH SCALES OF MODEL, ACQ_FUNC WHEN WE START TAKING RANGE OF COVARS INTO ACCOUNT
            - Add support for categorical and integer variables
            - Add support for adding name to each covariate
        :param covars (list of tuples): each entry (tuple) must contain (<initial_guess>, <min>, <max>) for each input
        variable. Currently only allows for continuous variables. Aspiration: Data type of input will be preserved and
        code be adapted to accommodate both integer and categorical variables.
        :return initial_guesses (torch.tensor size 1 X num_covars) (first row of design matrix)
        :return bounds (torch.tensor size 2 X num_covars)
        """

        # verify datatype of covars (pending)

        # extract initial guesses
        guesses = [[g[0] for g in covars]]

        # bounds
        lower_bounds = [g[1] for g in covars]
        upper_bounds = [g[2] for g in covars]

        return (
            torch.tensor(guesses, device=self.device, dtype=self.dtype),
            torch.tensor(
                [lower_bounds, upper_bounds], device=self.device, dtype=self.dtype
            ),
        )

    def __initialize_best_response(self):
        """
        initialize best response. If no data present sets to None, otherwise identifies best response up to each data
        point.
        """

        self.covars_best_response_value = None
        self.best_response_value = None
        first = True

        if self.train_Y is not None:
            for it in range(1, self.train_Y.shape[0] + 1):
                # for monkeypatching during unit testing to work, it is required that the full path is added to this
                # import (monkeypatching this function)
                max_X, max_Y = creative_project._max_response.find_max_response_value(
                    self.train_X[:it, :], self.train_Y[:it]
                )

                if first:
                    self.covars_best_response_value = max_X
                    self.best_response_value = max_Y
                    first = False
                else:
                    self.covars_best_response_value = torch.cat(
                        (self.covars_best_response_value, max_X), dim=0
                    )
                    self.best_response_value = torch.cat(
                        (self.best_response_value, max_Y), dim=0
                    )

    def __initialize_training_data(self, train_X, train_Y):
        """
        determine whether training data and response data have been added as part of initialization of class instance.
        Set attribute "start_from_guess"
        :param train_X (torch.tensor): design matrix of covariates (batch_shape X num_obs X num_training_features OR num_obs X num_training_features)
        :param train_Y (torch.tensor): observations (batch_shape X num_obs X num_output_models [allows for batched models] OR num_obs X num_output_models)
        :return (creates class attributes):
            - start_from_guess (bool): determines whether can start from provided training data (False if no/invalid/inconsistent data provided)
            - train_X (tensor): design matrix of covariates
            - train_Y (tensor): corresponding observations
            - proposed_X (tensor): the covariates proposed for analysis. Set to train_X
        """

        # bool attribute used in 'ask'-method to determine whether training data has been provided
        self.start_from_guess = True

        # initialize
        self.proposed_X = None
        self.train_X = None
        self.train_Y = None

        # only take action if no iterations have been taken in model
        if (self.model["covars_sampled_iter"] == 0) & (self.model["response_sampled_iter"] == 0):

            # check if data has been supplied via kwargs, otherwise train_X, train_Y will be None
            if (train_X is not None) & (train_Y is not None):

                # validate that provided training data has same number of rows in 'train_X' and 'train_Y' and that
                # number of variables in 'train_X' correspond to number variables provided in 'covars' (under class
                # instance initialization)
                if self._Validators__validate_training_data(train_X, train_Y):
                    # set flag attribute indicating data is acceptable for hotstart
                    self.start_from_guess = False

                    # update stored data
                    self.proposed_X = train_X
                    self.train_X = train_X
                    self.train_Y = train_Y