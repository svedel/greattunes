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
        :param covars_array (torch.tensor, pandas dataframe, numpy array)
        :param
            - state of initialized class:
                - self.initial_guess (torch.tensor)
        :return valid (bool)
        """

        valid = False

        if covars_array.shape[0] == self.initial_guess.shape[1]:
            valid = True

        return valid
