import torch
import warnings
import numpy as np


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

                # the case where the number of rows (observations) are not the same for train_X and train_Y
                else:
                    raise Exception(
                        "greattunes._validators.Validators.__validate_training_data: The number of"
                        " rows (observations) in provided 'train_X' ("
                        + str(train_X.size()[0])
                        + ") is"
                        " not the same as for `train_Y` ("
                        + str(train_Y.size()[0])
                        + ") as it should be."
                    )

        return valid

    def __validate_num_covars(self, covars_array):
        """
        validate that entries in "covars_array" is equal to number of covars provided to "covars" during
        class instance initialization of TuneSession (from greattunes.__init__.py)
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
        validate that covars is a list of tuples of types (float, int, str)
        :param covars (list of tuples): object, only accepted if covars is list of tuples of types (int, float, str)
        :return: valid (bool
        """

        valid = False

        if covars is None:
            raise ValueError(
                "greattunes.greattunes._validators.Validator.__validate_covars: covars is None"
            )

        if not isinstance(covars, list):
            raise TypeError(
                "greattunes.greattunes._validators.Validator.__validate_covars: covars is not list "
                "of tuples (not list)"
            )

        for entry in covars:
            if not isinstance(entry, tuple):
                raise TypeError(
                    "greattunes.greattunes._validators.Validator.__validate_covars: entry in covars list is not "
                    "tuple"
                )

        for entry in covars:
            for el in entry:
                if not isinstance(el, (float, int, str)):
                    raise TypeError(
                        "greattunes.greattunes._validators.Validator.__validate_covars: tuple element "
                        + str(el)
                        + " in covars list is neither of type float, int or str"
                    )

        valid = True

        return valid

    @staticmethod
    def __validate_num_entries_covar_tuples(covars, covars_tuple_datatypes):
        """
        ensures that the correct number of entries are present in the tuples of covars. covars_tuple_datatypes contains
        the tuple datatypes as determined by greattunes._initializers.__determine_tuple_datatype
        :param covars (list of tuples): object, only accepted if covars is list of tuples of types (int, float, str)
        :param covars_tuple_datatypes (list of types): types assigned to each tuple, e.g. by
        greattunes._initializers.__determine_tuple_datatype
        :return valid (bool)
        """

        # initialize
        valid = False

        # ensure the same number of entries
        assert len(covars) == len(covars_tuple_datatypes), (
            "greattunes._validators.__validate_num_entries_covar_tuples: dimension mismatch between the number of tuples in 'covars' and the number of datatype decisions in 'covars_tuple_datatypes'. 'covars' has length "
            + str(len(covars))
            + " while 'covars_tuple_datatype' has length "
            + str(len(covars_tuple_datatypes))
        )

        # check for compliance
        # int, float types must have exactly 3 entries; str must have at least 1 entry
        for i in range(len(covars)):
            if covars_tuple_datatypes[i] in {int, float}:
                assert len(covars[i]) == 3, (
                    "greattunes._validators.__validate_num_entries_covar_tuples: tuple entries of types (int, float) must have 3 entries. This is not the case for the entry "
                    + str(covars[i])
                )
            elif covars_tuple_datatypes[i] in {str}:
                assert len(covars[i]) > 0, (
                    "greattunes._validators.__validate_num_entries_covar_tuples: tuple entries of types str (categorical variables) must have at least 1 entry. This is not the case for the entry "
                    + str(covars[i])
                )

        # update output parameter if no errors thrown so far
        valid = True

        return valid

    @staticmethod
    def __validate_covars_dict_of_dicts(covars):
        """
        ensures that 'covars' (a dict of dicts determining the covariates for an optimization) have the right data
        types and the right content inside those data types. See doctring for
        greattunes._initializers.Initializers.__initialize_covars_dict_of_dicts for full description of correct
        format for covars

        for categorical variables also checks that the initial guess 'guess' is part of all potential outcomes for that
        variable, and adds if not the case

        :param covars (dict of dicts): each key-value pair in outer dict describes a covariate, with the key being the
        name used for this covariate. Each covariate is defined by a dict; for covariates of types 'int' and 'float'
        these must contain entries 'guess' (initial guess for value of covariate), 'min', 'max' and 'type' (must itself
        be among {int, float, str}); for categorical variables (use type 'str' to identify these), the required
        elements in the dict are 'guess', 'options' and 'type', where the middle one is a set of all possible values of
        the categorical variable and 'type' is the data type and must be among int, float and str. See doctring for
        greattunes._initializers.Initializers.__initialize_covars_dict_of_dicts for full description of correct
        format for covars
        :return valid (bool):
        :return covars_out (dict of dicts): input covars with any updates to categorical variables
        """

        # initialize
        valid = False

        # check that content is dicts
        list_content_types = [type(i) for i in list(covars.values())]
        if not set(list_content_types) == {dict}:
            raise Exception(
                "greattunes._validators.Validators.__validate_covars_dict_of_dicts: 'covars' provided as "
                "part of class initialization must be either a list of tuples or a dict of dicts. Current provided is "
                "a dict containing data types " + str(set(list_content_types)) + "."
            )

        # check that each has the right elements
        for key in covars.keys():
            # makes sure 'guess' is provided
            if "guess" not in covars[key].keys():
                raise Exception(
                    "greattunes._validators.Validators.__validate_covars_dict_of_dicts: key 'guess' "
                    "missing for covariate '"
                    + str(key)
                    + "' (covars['"
                    + str(key)
                    + "']="
                    + str(covars[key])
                    + ")."
                )

            # makes sure data type is provided
            if "type" not in covars[key].keys():
                raise Exception(
                    "greattunes._validators.Validators.__validate_covars_dict_of_dicts: key 'type' missing "
                    "for covariate '"
                    + str(key)
                    + "' (covars['"
                    + str(key)
                    + "']="
                    + str(covars[key])
                    + ")."
                )

            else:
                # warning if types beyond int, float, str are provided
                if covars[key]["type"] not in {int, float, str}:
                    warnings.warn(
                        "greattunes._validators.Validators.__validate_covars_dict_of_dicts: key "
                        + str(key)
                        + " will be ignored because its data type '"
                        + str(covars[key]["type"])
                        + "' is not "
                        "among supported types {int, float, str}."
                    )

                # checks for int, float
                if covars[key]["type"] in {int, float}:
                    if "min" not in covars[key].keys():
                        raise Exception(
                            "greattunes._validators.Validators.__validate_covars_dict_of_dicts: key 'min' "
                            "missing for covariate '"
                            + str(key)
                            + "' (covars['"
                            + str(key)
                            + "']="
                            + str(covars[key])
                            + ")."
                        )
                    if "max" not in covars[key].keys():
                        raise Exception(
                            "greattunes._validators.Validators.__validate_covars_dict_of_dicts: key 'max' "
                            "missing for covariate '"
                            + str(key)
                            + "' (covars['"
                            + str(key)
                            + "']="
                            + str(covars[key])
                            + ")."
                        )
                elif covars[key]["type"] == str:

                    # checks whether "options" (set of categorical options) is present
                    if "options" not in covars[key].keys():
                        raise Exception(
                            "greattunes._validators.Validators.__validate_covars_dict_of_dicts: key "
                            "'options' missing for covariate '"
                            + str(key)
                            + "' (covars['"
                            + str(key)
                            + "']="
                            + str(covars[key])
                            + ")."
                        )

                    # add value from "guess" to list of options if not already present
                    if covars[key]["guess"] not in covars[key]["options"]:
                        covars[key]["options"].add(covars[key]["guess"])

        valid = True
        covars_out = covars

        return valid, covars_out

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

    def __continue_iterating_rel_tol_conditions(self, rel_tol, rel_tol_steps):
        """
        determine whether 'rel_tol'-conditions are satisfied by comparing the best candidate (self.best_response_value)
        at this and previous steps. In case the 'rel_tol'-conditions are met, this method returns false.

        The 'rel_tol'-conditions for stopping are:
            - 'rel_tol' and 'rel_tol_steps' are both None: nothing happens, and this method returns True at all
            iterations
            - 'rel_tol' is not None and 'rel_tol_steps' is None: in this case, stop at the first iteration where the
            relative difference between self.best_response_value at this and the preceding iteration are below the
            value set by 'rel_tol' (stop by returning False)
            -  'rel_tol' is not None and 'rel_tol_steps' is not None: stop when the relative difference of
            self.best_response_value at the current and the 'rel_tol_steps' preceding steps are all below 'rel_tol'
            (return False)

        :param rel_tol (float): limit of relative difference of self.best_response_value taken as convergence
        :param rel_tol_steps (int): number of consecutive steps in which the improvement in self.best_response_value
        is below 'rel_tol' before stopping
        :return: continue_iterating (bool)
        """

        continue_iterating = True

        if (rel_tol is not None) & (rel_tol_steps is None):

            # leverage the functionality built for rel_tol_steps > 1
            rel_tol_steps = 1

        if (
            (rel_tol is not None)
            & (rel_tol_steps is not None)
            & (self.best_response_value is not None)
        ):

            # only proceed if at least 'rel_tol_steps' iterations have been completed
            if self.best_response_value.size()[0] > rel_tol_steps:

                # === Build list of relative differences ===
                # first build tensor with the last rel_tol_steps entries in self.best_response_value and the last
                # rel_tol_steps+1 entries
                tmp_array = torch.cat(
                    (
                        self.best_response_value[-(rel_tol_steps + 1) : -1],
                        self.best_response_value[-(rel_tol_steps):],
                    ),
                    dim=1,
                ).numpy()

                # calculate the relative differences
                tmp_rel_diff = (
                    np.diff(tmp_array, axis=1)
                    / self.best_response_value[-(rel_tol_steps):].numpy()
                )

                # determine if all below 'rel_tol'
                below_rel_tol = [
                    rel_dif[0] < rel_tol for rel_dif in tmp_rel_diff.tolist()
                ]

                # only accept if the relative difference is below 'rel_tol' for all steps
                if sum(below_rel_tol) == rel_tol_steps:
                    continue_iterating = False

        return continue_iterating

    @staticmethod
    def __validate_model_acceptable(model, model_list):

        valid = False

        if model not in model_list:
            raise Exception(
                "greattunes._validators.__validate_model_acceptable:"
                + model
                + " is not among approved options ("
                + ", ".join(model_list)
                + ")"
            )
        else:
            valid = True

        return valid
