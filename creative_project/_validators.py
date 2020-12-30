import torch


class Validators:
    """
    validator functions used to initialize data structures etc. These functions are to remain private, i.e.
    non-accessible to the user.

    These methods are kept in a separate class to maintain that they be private in the main
    class (that class being defined in __init__.py.
    """

    def __validate_training_data(self, train_X, train_Y):
        """
        validate provided training data is of compatible length
        :parameter train_X (torch.tensor, size: batch_shape X num_obs X num_training_features OR num_obs X
        num_training_features)
        :parameter train_Y (torch.tensor, size batch_shape X num_obs X num_output_models [allows for batched models]
        OR num_obs X num_output_models)
        :return valid (bool)
        """

        valid = False

        # validate training data present
        if (train_X is not None) & (train_Y is not None):

            # validate data type (tensors are set to type torch.DoubleTensor by setting dtype=torch.double)
            if isinstance(train_X, torch.DoubleTensor) & isinstance(
                train_Y, torch.DoubleTensor
            ):

                # validate same number of rows
                if train_X.shape[0] == train_Y.shape[0]:

                    # validate train_X number of covariates correspond to covars
                    if self.__validate_num_covars(
                        train_X
                    ):  # train_X.shape[0] == self.initial_guess.shape[0]:

                        valid = True

        return valid

    def __validate_num_covars(self, covars_array):
        """
        validate that entries in "covars_array" is equal to number of covars provided to "covars" during
        class instance initialization of CreativeProject (from creative_project.__init__.py)
        :param covars_array (torch.tensor, pandas dataframe, numpy array; shape needs to be
        num_observations X num_covariates)
        :param
            - state of initialized class:
                - self.initial_guess (torch.tensor)
        :return valid (bool)
        """

        valid = False

        # with one column per covariate in covars_array, and one column per covariate in initial_guess, makes sure that
        # same amount of covariates present in both
        if covars_array is not None:
            if len(covars_array.shape) > 1:
                if covars_array.shape[1] == self.initial_guess.shape[1]:
                    valid = True

        return valid

    def __validate_covars(self, covars):
        """
        validate that covars is a list of tuples of floats
        :param covars: object, only accepted if covars is list of tuples of floats
        :return: valid (bool
        """

        valid = False

        if covars is None:
            raise ValueError(
                "kre8_core.creative_project._validators.Validator.__validate_covars: covars is None"
            )

        if not isinstance(covars, list):
            raise TypeError(
                "kre8_core.creative_project._validators.Validator.__validate_covars: covars is not list "
                "of tuples (not list)"
            )

        for entry in covars:
            if not isinstance(entry, tuple):
                raise TypeError(
                    "kre8_core.creative_project._validators.Validator.__validate_covars: entry in covars list is not "
                    "tuple"
                )

        for entry in covars:
            for el in entry:
                if not isinstance(el, (float, int)):
                    raise TypeError(
                        "kre8_core.creative_project._validators.Validator.__validate_covars: tuple element "
                        + str(el)
                        + " in covars list is neither of type float or int"
                    )

        valid = True

        return valid

    def __validate_num_response(self, response_array):
        """
        validate that there is only one response per timepoint in "response_array"
        :param covars_array (torch.tensor, pandas dataframe, numpy array; shape needs to be
        num_observations X num_covariates)
        :return valid (bool)
        """

        valid = False

        # make sure single column
        if response_array is not None:
            if len(response_array.shape) > 1:
                if response_array.shape[1] == 1:
                    valid = True

        return valid
