import math
import torch
import warnings
from creative_project._validators import Validators
from creative_project.utils import DataSamplers


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
        if self._Validators__validate_covars(covars=covars):

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
                max_X, max_Y = self._find_max_response_value(
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
        :param train_X (torch.tensor): design matrix of covariates (batch_shape X num_obs X num_training_features OR
        num_obs X num_training_features)
        :param train_Y (torch.tensor): observations (batch_shape X num_obs X num_output_models
        [allows for batched models] OR num_obs X num_output_models)
        :return (creates class attributes):
            - start_from_guess (bool): determines whether can start from provided training data (False if
            no/invalid/inconsistent data provided)
            - train_X (tensor): design matrix of covariates
            - train_Y (tensor): corresponding observations
            - proposed_X (tensor): the covariates proposed for analysis. Set to train_X
        """

        # bool attribute used in 'ask'-method to determine whether training data has been provided
        #self.start_from_guess = True

        # initialize
        self.proposed_X = None
        self.train_X = None
        self.train_Y = None

        # only take action if no iterations have been taken in model
        if (self.model["covars_sampled_iter"] == 0) & (
            self.model["response_sampled_iter"] == 0
        ):

            # check if data has been supplied via kwargs, otherwise train_X, train_Y will be None
            if (train_X is not None) & (train_Y is not None):

                # validate that provided training data has same number of rows in 'train_X' and 'train_Y' and that
                # number of variables in 'train_X' correspond to number variables provided in 'covars' (under class
                # instance initialization)
                if self._Validators__validate_training_data(train_X, train_Y):
                    # set flag attribute indicating data is acceptable for hotstart
                    self.start_from_guess = False

                    # update stored data
                    self.train_X = train_X
                    self.train_Y = train_Y

                    # initialize proposed_X with empty (i.e. 0 to machine precision) since we have no proposals for
                    # train_X
                    self.proposed_X = torch.empty(train_X.size())

    def __initialize_random_start(self, random_start, num_initial_random, random_sampling_method):
        """
        set details for how to randomize at start
        :param random_start (bool): input from user (set during CreativeProject.__init__)
        :param num_initial_random (int/None): provided as kwarg input from user (set during CreativeProject.__init__)
        :param random_sampling_method (str/None): sampling method for random points. Options: "random" and "latin_hcs"
        (latin hypercube sampling). Provided as kwarg input from user (set during CreativeProject.__init__)
        :param (attributes)
            - self.train_X (tensor/None): user-provided data for covariates. Only present if data passes validation
            - self.train_Y (tensor/None): user-provided data for response. Only present if data passes validation
        :return:
            - self.num_initial_random_points (int): number of initial random datapoints. If set to 0, there's no
            random start
            - self.random_sampling_method (str): sampling method for random points. Options: "random" and "latin_hcs"
        (latin hypercube sampling).
        """

        # control that 'random_sampling_method' is part of acceptable options
        SAMPLING_METHODS_LIST = [func for func in dir(DataSamplers) if callable(getattr(DataSamplers, func)) and not func.startswith("__")]
        if (random_sampling_method is not None)&(random_sampling_method not in SAMPLING_METHODS_LIST):
            raise Exception("creative_project._initializers.Initializers.__initialize_random_start: The parameter "
                            "'random_sampling_method' is not among allowed values ('" + "', '".join(SAMPLING_METHODS_LIST) + "').")

        # set sampling method
        if random_sampling_method is None:
            self.random_sampling_method = "latin_hcs"  # default to latin hypercube sampling if not specified
        else:
            self.random_sampling_method = random_sampling_method

        # split: train_X, train_Y present
        if (self.train_X is not None)&(self.train_Y is not None):

            # CASE 1: train_X, train_Y present; random_start = False
            if not random_start:

                # set 0 initial random sampling points
                self.num_initial_random_points = 0
                # in this special case, do not set any sampling method (since won't use)
                self.random_sampling_method = None

            # CASE 2: train_X, train_Y present; random_start = True
            else:

                # set number of initial points or make own guess
                self.num_initial_random_points = self.determine_number_random_samples()
                if num_initial_random is not None:
                    self.num_initial_random_points = num_initial_random

        # split: train_X, train_Y NOT present
        else:

            # CASE 3: train_X, train_Y NOT present; random_start = False
            if not random_start:
                self.num_initial_random_points = self.determine_number_random_samples()
                # TODO: when adding logging, switch to setting this warning via logging message with level "warn", see
                # here: "As an alternative, especially for standalone applications, consider the logging module. It can
                # log messages having a level of debug, info, warning, error, etc. Log messages having a level of
                # warning or higher are by default printed to stderr."
                warnings.warn("Inconsistent settings for optimization initialization: No initial data provided via "
                              "'train_X' and 'train_Y' but also 'random_start' is set to 'False'. Adjusting to start"
                              " with " + str(self.num_initial_random_points) + " random datapoints")

            # CASE 4: train_X, train_Y NOT present; random_start = True
            else:

                self.num_initial_random_points = self.determine_number_random_samples()
                if num_initial_random is not None:
                    self.num_initial_random_points = num_initial_random

    def determine_number_random_samples(self):
        """
        estimates number of random samples to initiate an optimization run. Rule of thumb is to do sqrt(d) initial
        random steps for a problem with d covariates
        :param
            - self.__covars (list of tuples): the user-provided details on covariates. Here using only the length of
            the list, which gives the number of covariates
        :return: num_random (int)
        """

        NUM_COVARS = len(self.__covars)
        num_random = int(round(math.sqrt(NUM_COVARS), 0))

        return num_random
