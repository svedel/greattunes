import torch
from ._initializers import Initializers


class CreativeProject(Initializers):
    """
    Class of user-facing functionality to handle creative projects.
    """

    # Initialize class instance
    def __init__(self, covars, model="SingleTaskGP", acq_func="EI", **kwargs):
        """
        defines the optimization model. Uses BoTorch underneath the hood, so any BoTorch model and aqcuisition function
        can be used. "model" and "aqc_func" must follow BoTorch nomeclature. A list of custom models is also
        available/under development to be included.
        :param covars (list of tuples): each entry (tuple) must contain (<initial_guess>, <min>, <max>) for each input
        variable. Currently only allows for continuous variables. Aspiration: Data type of input will be preserved and
        code be adapted to accommodate both integer and categorical variables.
        :param model str: sets surrogate model (currently allow "SingleTaskGP" and "Custom")
        :param acq_func (str): sets acquisition function (currently allows only "EI")
        :param sampling_type (str): "manual"/"functions" determines whether to get data via manual input from prompt or
        from functions
        :kwargs
            - train_X (torch.tensor of dtype=torch.double): design matrix of covariates (batch_shape X num_obs X
            num_training_features OR num_obs X num_training_features)
            - train_Y (torch.tensor of dtype=torch.double): observations (batch_shape X num_obs X num_output_models
            [allows for batched models] OR num_obs X num_output_models)
            - kernel (str): kernel used to define custom GP models (not currently in use)
            - nu (float): kernel parameter for Matern kernel
        """

        # Computational settings
        # determines computation device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # data type for tensors
        self.dtype = torch.double

        # Data initialization
        # initialize the data (initial guess) and bounds on covars
        (
            self.initial_guess,
            self.covar_bounds,
        ) = self._Initializers__initialize_from_covars(covars)

        # define the model
        self.model = {
            "model_type": model,
            "model": None,
            "likelihood": None,
            "loglikelihood": None,
            "covars_proposed_iter": 0,
            "covars_sampled_iter": 0,
            "response_sampled_iter": 0,
        }

        # define kernel --- THIS SHOULD BE DONE MORE ELEGANTLY. If not present as input, these will be set to None
        self.nu = kwargs.get("nu")  # get value of nu for Matern kernel

        # define acquisition function
        self.acq_func = {
            "type": acq_func,
            "object": None,
        }

        # define sampling functions
        # initialize as manual. Will be updated if method "auto" is used (use when sampling function known)
        sampling_type = "manual"
        self.sampling = {"method": sampling_type, "response_func": None}

        # initialize data for training and storage:
        #     - self.train_X (num_covars X num_obs): observed design matrix
        #     - self.train_Y (1 X num_obs): observed response
        #     - self.proposed_X (num_covars X num_obs): matrix of proposed new covars datapoints to sample at each
        #     iteration
        # grab training data if passed through kwargs. If any is not present 'train_X' and/or 'train_Y' will be set to
        # None
        # !!! ADD STANDARDIZATION !!!
        # !!! CONSIDER CREATING A DICT OF DATA INSTEAD OF THREE SEPARATE ATTRIBUTES !!!
        self.__initialize_training_data(
            train_X=kwargs.get("train_X"), train_Y=kwargs.get("train_Y")
        )

        # best observed candidate (best response) [self.best_response_value 1 X num_obs tensor], together with
        # corresponding covariates [self.covariates_best_response_value num_covars X num_obs tensor]
        # self.__initialize_best_response()
