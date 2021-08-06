from collections import OrderedDict

import math
import pandas as pd
import warnings

import torch

from greattunes._validators import Validators
from greattunes.utils import DataSamplers
from greattunes.data_format_mappings import (
    tensor2pretty_covariate,
    pretty2tensor_covariate,
    tensor2pretty_response,
    pretty2tensor_response,
)


class Initializers(Validators):
    """
    initializer functions used to initialize data structures etc. These functions are to remain private, i.e.
    non-accessible to the user.

    These methods are kept in a separate class to maintain that they be private in the main
    class (that class being defined in __init__.py).

    For simplicity is developed as child of Validators class (from greattunes._validators), making all validator
    methods available inside Initializers
    """

    def __initialize_from_covars(self, covars):
        """
        initialize covariates for modeling from the input "covars" provided by user. This method sets the number of
        covariates and their data type, which are the covariates available for the optimization.

        The input "covars" can be either in the form of a list of tuples (one tuple per covariate) or in the form of a
        dict of dicts (one embedded dict per covariate). For the former, each tuple defines the range the variable can
        vary within, and the data type of the covariate (integer, continuous or categorical) is automatically derived
        from the content of each tuple. Integer and continous variables must have 3 entries in the tuple as follows
        (<guessed_value>,<minimum_value>,<maximum_value>); categorical variables can have any number of elements as
        long as at least one element is provided.

            Example:

                covars = [
                            (1, 0, 2),  # will be taken as INTEGER (type: int)
                            (1.0, 0.0, 2.0),  # will be taken as CONTINUOUS (type: float)
                            (1, 0, 2.0),  # will be taken as CONTINUOUS (type: float)
                            ("red", "green", "blue", "yellow"),  # will be taken as CATEGORICAL (type: str)
                            ("volvo", "chevrolet", "ford"),  # will be taken as CATEGORICAL (type: str)
                            ("sunny", "cloudy"),  # will be taken as CATEGORICAL (type: str)
                        ]

        For the second case where "covars" is a dict of dicts each key-value pair in outer dict describes a covariate,
        with the key being the name used for this covariate. Each covariate is defined by a dict; for covariates of
        types 'int' and 'float' these must contain entries 'guess' (initial guess for value of covariate), 'min', 'max'
        and 'type' (must itself be among {int, float, str}); for categorical variables (use type 'str' to identify
        these), the required elements in the dict are 'guess', 'options' and 'type', where the middle one is a set of
        all possible values of the categorical variable and 'type' is the data type and must be among int, float and
        str

            Example:

            covars = {
                        'variable1:  # type: integer
                            {
                                'guess': 1,
                                'min': 0,
                                'max': 2,
                                'type': int,
                            },
                        'variable2':  # type: continuous (float)
                            {
                                'guess': 12.2,
                                'min': -3.4,
                                'max': 30.8,
                                'type': float,
                            },
                        'variable3':  # type: categorical (str)
                            {
                                'guess': 'red',
                                'options': {'red', 'blue', 'green'},
                                'type': str,
                            }
                        }

        TODO:
            - ALSO CONSIDER CHANGING LENGTH SCALES OF MODEL, ACQ_FUNC WHEN WE START TAKING RANGE OF COVARS INTO ACCOUNT

        :param covars (list of tuples OR dict of dicts):
        For list of tuples: each entry (tuple) must contain (<initial_guess>, <min>, <max>) for each input
        variable. Currently only allows for continuous variables. Aspiration: Data type of input will be preserved and
        code be adapted to accommodate both integer and categorical variables.
        For dict of dicts: each key-value pair in outer dict describes a covariate, with the key being the
        name used for this covariate. Each covariate is defined by a dict; for covariates of types 'int' and 'float'
        these must contain entries 'guess' (initial guess for value of covariate), 'min', 'max' and 'type' (must itself
        be among {int, float, str}); for categorical variables (use type 'str' to identify these), the required
        elements in the dict are 'guess', 'options' and 'type', where the middle one is a set of all possible values of
        the categorical variable and 'type' is the data type and must be among int, float and str
        :return initial_guesses (torch.tensor size 1 X num_covars) (first row of design matrix)
        :return bounds (torch.tensor size 2 X num_covars)
        :return (attributes added):
            - covar_details (dict of dicts): contains a dict with details about each covariate wuth initial guess and
            range as well as information about which column it is mapped to in train_X dataset used by the Gaussian
            process model behind the scenes and data type of covariate. Includes one-hot encoding for categorical
            variables. For one-hot encoded categorical variables use the naming convention
            <covariate name>_<option name>
            - GP_kernel_mapping_covar_identification (list of dicts): reduced version of 'covar_details' containing
            only data type and mapped columns information
            - covar_mapped_names (list): names of mapped covariates
            - total_num_covars (int): total number of columns created for backend train_X dataset used by Gaussian
            processes
        """

        # list of tuples provided as covars
        if type(covars) == list:

            # generate 'covar_details', 'GP_kernel_mapping_covar_identification', 'total_num_covars',
            # 'covar_mapped_names'
            self.__initialize_covars_list_of_tuples(covars=covars)

        # dictionary provided
        elif type(covars) == dict:

            # generate 'covar_details', 'GP_kernel_mapping_covar_identification', 'total_num_covars',
            # 'covar_mapped_names'
            self.__initialize_covars_dict_of_dicts(covars=covars)

        # other data types, throws error
        else:
            raise Exception(
                "greattunes._initializers.Initializers.__initialize_from_covars: provided 'covars' is"
                " of type "
                + str(type(covars))
                + " but must be of types {'list', 'dict'}."
            )

        # === extract initial guesses and covariate bounds ===

        # stuff to save as attributes
        ig_tmp = [0 for i in range(self.total_num_covars)]
        lb_tmp = [0 for i in range(self.total_num_covars)]
        ub_tmp = [0 for i in range(self.total_num_covars)]
        for i in self.covar_details:

            tmp_col = self.covar_details[i]["columns"]

            # for int, float has only single column
            if self.covar_details[i]["type"] in {int, float}:
                ig_tmp[tmp_col] = self.covar_details[i]["guess"]
                lb_tmp[tmp_col] = self.covar_details[i]["min"]
                ub_tmp[tmp_col] = self.covar_details[i]["max"]

            # for str have multiple columns
            elif self.covar_details[i]["type"] == str:

                # first one-hot column is the guess (we start at 100 % committed guess to this column)
                # rest of one-hot columns are initialized at 0, so no need to update
                ig_tmp[tmp_col[0]] = 1.0

                # set range for all one-hot columns as [0; 1]
                for j in tmp_col:
                    lb_tmp[j] = 0.0
                    ub_tmp[j] = 1.0

        initial_guesses = torch.tensor([ig_tmp], device=self.device, dtype=self.dtype)
        covar_bounds = torch.tensor(
            [lb_tmp, ub_tmp], device=self.device, dtype=self.dtype
        )

        return initial_guesses, covar_bounds

    @staticmethod
    def __determine_tuple_datatype(x_tuple):
        """
        determines which data type to assign to the content of the tuple 'x_tuple'. Elements must be only of types
        (int, float, str), and the method assigns according to the following priorities
        - if any entry is a str -> all entries cast as str
        - if any entry is a float -> all entries cast as float
        that means, that if any entry is a str the content of that tuple is considered str; while, if any entry in a
        tuple is a float, then elements of that tuple are considered float (even if they are originally int). A tuple
        is considered to contain only int data type if all entries have the data type int.

        :param x_tuple (tuple of int, float or str): it is to the elements of this tuple we are determining the type
        :return tuple_datatype (type):
        """

        # initialize
        tuple_datatype = float

        # types in provided tuple
        types = [type(i) for i in x_tuple]

        # check that only float and int types provided
        for t in set(types):
            if t not in {int, float, str}:
                raise Exception(
                    "greattunes._initializers.Initialzer.__determine_tuple_datatype: individual covariates "
                    "provided via tuples can only be of types ('float', 'int', 'str') but was provided "
                    + str(t)
                )

        # if any str available in tuple treat as categorical
        if str in types:
            tuple_datatype = str

        # special case if only int data types provided
        elif set(types) == {int}:
            tuple_datatype = int

        return tuple_datatype

    def __initialize_covars_list_of_tuples(self, covars):
        """
        investigating the case when covars is a list of tuples, ensures only acceptable number of entries and data
        types and creates the needed data structures 'covar_details', 'GP_kernel_mapping_covar_identification' and
        'total_num_covars' as attributes.

        assumes that covars is of type list of tuples

        :param covars (list of tuples): each entry (tuple) must contain (<initial_guess>, <min>, <max>) for each input
        variable. Currently only allows for continuous variables. Aspiration: Data type of input will be preserved and
        code be adapted to accommodate both integer and categorical variables.

            Example:

                covars = [
                            (1, 0, 2),  # will be taken as INTEGER (type: int)
                            (1.0, 0.0, 2.0),  # will be taken as CONTINUOUS (type: float)
                            (1, 0, 2.0),  # will be taken as CONTINUOUS (type: float)
                            ("red", "green", "blue", "yellow"),  # will be taken as CATEGORICAL (type: str)
                            ("volvo", "chevrolet", "ford"),  # will be taken as CATEGORICAL (type: str)
                            ("sunny", "cloudy"),  # will be taken as CATEGORICAL (type: str)
                        ]

        :return as attributes
            - covar_details (dict of dicts): contains a dict with details about each covariate wuth initial guess and
            range as well as information about which column it is mapped to in train_X dataset used by the Gaussian
            process model behind the scenes and data type of covariate. Includes one-hot encoding for categorical
            variables. For one-hot encoded categorical variables use the naming convention
            <covariate name>_<option name>
            - GP_kernel_mapping_covar_identification (list of dicts): reduced version of 'covar_details' containing
            only data type and mapped columns information
            - covar_mapped_names (list): names of mapped covariates
            - total_num_covars (int): total number of columns created for backend train_X dataset used by Gaussian
            processes
            - sorted_pandas_columns (list): ordered list of names of columns for pretty data (pandas)
        """

        # first ensure the format of 'covars' is as expected
        assert self._Validators__validate_covars(covars=covars)

        # assign names to all columns. Starting from tuples we don't have names assigned, so creating own
        covar_names = ["covar" + str(i) for i in range(len(covars))]

        # determine which datatype to assign content of each tuple and ensures these are acceptable
        covar_types = [self.__determine_tuple_datatype(tpl) for tpl in covars]
        assert self._Validators__validate_num_entries_covar_tuples(
            covars=covars, covars_tuple_datatypes=covar_types
        )

        # initialize attributes
        GP_kernel_mapping_covar_identification = []
        covar_details = OrderedDict()
        covar_mapped_names = []

        # loop through each entry in list of tuples and builds out 'covar_details',
        # 'GP_kernel_mapping_covar_identification', 'covar_mapped_names' and total count with the right information
        column_counter = 0

        for i in range(len(covars)):

            # case where data type is int or float. only a single column needed in train_X dataset for covariates used
            # behind the scenes for Gaussian process modeling
            if covar_types[i] in {int, float}:
                covar_details[covar_names[i]] = {
                    "guess": covars[i][0],
                    "min": covars[i][1],
                    "max": covars[i][2],
                    "type": covar_types[i],
                    "columns": column_counter,
                    "pandas_column": i,
                }

                # update book keeping
                if covar_types[i] == int:
                    GP_kernel_mapping_covar_identification += [
                        {"type": int, "column": [column_counter]}
                    ]
                else:
                    GP_kernel_mapping_covar_identification += [
                        {"type": float, "column": [column_counter]}
                    ]

                column_counter += 1
                covar_mapped_names += [covar_names[i]]

            # special situation for categorical variables (data type is str).
            # in this case builds the options and does the one-hot encoding mapping to split the categorical variable
            # into a set of new continuous variables containing one new continuous variable per categorical option.
            # For one-hot encoded categorical variables use the naming convention <covariate name>_<option name>
            elif covar_types[i] == str:

                # determines the names of the one-hot encoded continuous variables
                num_opts = len(covars[i])
                opt_names = [covar_names[i] + "_" + j for j in covars[i]]

                covar_details[covar_names[i]] = {
                    "guess": covars[i][0],
                    "options": {
                        str(j) for j in covars[i]
                    },  # converts all entries to str if type is str
                    "type": covar_types[i],
                    "columns": [
                        j + column_counter for j in range(num_opts)
                    ],  # # sets mapped one-hot columns, names
                    "opt_names": opt_names,
                    "pandas_column": i,
                }

                # update book keeping
                GP_kernel_mapping_covar_identification += [
                    {
                        "type": str,
                        "column": [j + column_counter for j in range(num_opts)],
                    }
                ]

                column_counter += num_opts
                covar_mapped_names += [j for j in opt_names]

        # save attributes
        self.covar_details = covar_details
        self.GP_kernel_mapping_covar_identification = (
            GP_kernel_mapping_covar_identification
        )
        self.covar_mapped_names = covar_mapped_names
        self.total_num_covars = column_counter
        self.sorted_pandas_columns = covar_names

    def __initialize_covars_dict_of_dicts(self, covars):
        """
        investigating the case when covars is a dict of dicts, ensures only acceptable number of entries and data
        types and creates the needed data structures 'covar_details', 'GP_kernel_mapping_covar_identification' and
        'total_num_covars' as attributes.

        user-provided covars is extended with information about mapped column numbers (taking one-hot encoded
        categorical variables into consideration) to become 'covar_details'; present method assumes that covars is of
        type dict of dicts following the format of covar_details

        :param covars (dict of dicts): each key-value pair in outer dict describes a covariate, with the key being the
        name used for this covariate. Each covariate is defined by a dict; for covariates of types 'int' and 'float'
        these must contain entries 'guess' (initial guess for value of covariate), 'min', 'max' and 'type' (must itself
        be among {int, float, str}); for categorical variables (use type 'str' to identify these), the required
        elements in the dict are 'guess', 'options' and 'type', where the middle one is a set of all possible values of
        the categorical variable and 'type' is the data type and must be among int, float and str
            Example:

            covars = {
                        'variable1:  # type: integer
                            {
                                'guess': 1,
                                'min': 0,
                                'max': 2,
                                'type': int,
                            },
                        'variable2':  # type: continuous (float)
                            {
                                'guess': 12.2,
                                'min': -3.4,
                                'max': 30.8,
                                'type': float,
                            },
                        'variable3':  # type: categorical (str)
                            {
                                'guess': 'red',
                                'options': {'red', 'blue', 'green'},
                                'type': str,
                            }
                        }

        :return as attributes
            - covar_details (dict of dicts): contains a dict with details about each covariate wuth initial guess and
            range as well as information about which column it is mapped to in train_X dataset used by the Gaussian
            process model behind the scenes and data type of covariate. Includes one-hot encoding for categorical
            variables. For one-hot encoded categorical variables use the naming convention
            <covariate name>_<option name>
            - GP_kernel_mapping_covar_identification (list of dicts): reduced version of 'covar_details' containing
            only data type and mapped columns information
            - covar_mapped_names (list): names of mapped covariates
            - total_num_covars (int): total number of columns created for backend train_X dataset used by Gaussian
            processes
            - sorted_pandas_columns (list): ordered list of names of columns for pretty data (pandas)
        """

        # validate provided covars, for all categorical variables add 'guess' to 'options' in case missed
        valid, covars = self._Validators__validate_covars_dict_of_dicts(covars=covars)

        # assume that provided dict has right information except columns and opt_names in case of categorical variables
        covar_names = list(covars.keys())

        # initialize attributes
        GP_kernel_mapping_covar_identification = []
        covar_details = OrderedDict()
        for key, value in covars.items():
            covar_details[key] = value
        covar_mapped_names = []

        # loop through each entry in list of tuples and builds out 'covar_details',
        # 'GP_kernel_mapping_covar_identification', 'covar_mapped_names' and total count with the right information
        column_counter = 0
        original_column_counter = 0

        for key in covar_names:

            # case where data type is int or float. only a single column needed in train_X dataset for covariates used
            # behind the scenes for Gaussian process modeling
            if covar_details[key]["type"] in {int, float}:
                covar_details[key]["columns"] = column_counter
                covar_details[key]["pandas_column"] = original_column_counter

                # update book keeping
                if covar_details[key]["type"] == int:
                    GP_kernel_mapping_covar_identification += [
                        {"type": int, "column": [column_counter]}
                    ]
                else:
                    GP_kernel_mapping_covar_identification += [
                        {"type": float, "column": [column_counter]}
                    ]

                column_counter += 1
                original_column_counter += 1
                covar_mapped_names += [key]

            # special situation for categorical variables (data type is str).
            # in this case builds the options and does the one-hot encoding mapping to split the categorical variable
            # into a set of new continuous variables containing one new continuous variable per categorical option.
            # For one-hot encoded categorical variables use the naming convention <covariate name>_<option name>
            elif covar_details[key]["type"] == str:

                # determines the names of the one-hot encoded continuous variables
                num_opts = len(covar_details[key]["options"])
                opt_names = [key + "_" + j for j in covar_details[key]["options"]]

                # sets mapped one-hot encoded continuous columns and names
                covar_details[key]["columns"] = [
                    j + column_counter for j in range(num_opts)
                ]
                covar_details[key]["opt_names"] = opt_names
                covar_details[key]["pandas_column"] = original_column_counter

                # update book keeping
                GP_kernel_mapping_covar_identification += [
                    {
                        "type": str,
                        "column": [j + column_counter for j in range(num_opts)],
                    }
                ]

                # updates counters
                column_counter += num_opts
                original_column_counter += 1
                covar_mapped_names += [j for j in opt_names]

        # save attributes
        self.covar_details = covar_details
        self.GP_kernel_mapping_covar_identification = (
            GP_kernel_mapping_covar_identification
        )
        self.covar_mapped_names = covar_mapped_names
        self.total_num_covars = column_counter
        self.sorted_pandas_columns = covar_names

    def __initialize_best_response(self):
        """
        initialize best response. If no data present sets to None, otherwise identifies best response up to each data
        point.
        """

        # initialize tensor variables
        self.covars_best_response_value = None
        self.best_response_value = None

        # initialize pretty variables
        self.covars_best_response = None
        self.best_response = None

        first = True

        if self.train_Y is not None:
            for it in range(1, self.train_Y.shape[0] + 1):
                max_X, max_Y = self._find_max_response_value(
                    self.train_X[:it, :], self.train_Y[:it]
                )

                if first:

                    # tensor format
                    self.covars_best_response_value = max_X
                    self.best_response_value = max_Y

                    # pretty format
                    self.covars_best_response = tensor2pretty_covariate(
                        train_X_sample=max_X, covar_details=self.covar_details
                    )
                    self.best_response = tensor2pretty_response(train_Y_sample=max_Y)

                    first = False
                else:

                    # tensor format
                    self.covars_best_response_value = torch.cat(
                        (self.covars_best_response_value, max_X), dim=0
                    )
                    self.best_response_value = torch.cat(
                        (self.best_response_value, max_Y), dim=0
                    )

                    # pretty format
                    self.covars_best_response = self.covars_best_response.append(
                        tensor2pretty_covariate(
                            train_X_sample=max_X, covar_details=self.covar_details
                        )
                    )
                    self.best_response = self.best_response.append(
                        tensor2pretty_response(train_Y_sample=max_Y)
                    )

    def __initialize_training_data(self, train_X, train_Y):
        """
        determine whether training data and response data have been added as part of initialization of class instance.
        :param train_X (torch.tensor): design matrix of covariates (batch_shape X num_obs X num_training_features OR
        num_obs X num_training_features)
        :param train_Y (torch.tensor): observations (batch_shape X num_obs X num_output_models
        [allows for batched models] OR num_obs X num_output_models)
        :return (creates class attributes):
            - train_X (tensor): design matrix of covariates
            - train_Y (tensor): corresponding observations
            - proposed_X (tensor): the covariates proposed for analysis. Set to train_X
        """

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

                # convert to tensor if data provided in pandas framework
                if isinstance(train_X, pd.DataFrame) & isinstance(
                    train_Y, pd.DataFrame
                ):

                    train_X, _ = pretty2tensor_covariate(
                        x_pandas=train_X,
                        covar_details=self.covar_details,
                        covar_mapped_names=self.covar_mapped_names,
                        device=self.device,
                    )
                    train_Y = pretty2tensor_response(
                        y_pandas=train_Y, device=self.device
                    )

                # validate that provided training data has same number of rows in 'train_X' and 'train_Y' and that
                # number of variables in 'train_X' correspond to number variables provided in 'covars' (under class
                # instance initialization)
                if self._Validators__validate_training_data(train_X, train_Y):
                    # set flag attribute indicating data is acceptable for hotstart

                    # update stored data
                    self.train_X = train_X
                    self.train_Y = train_Y

                    # initialize proposed_X with empty (i.e. 0 to machine precision) since we have no proposals for
                    # train_X
                    self.proposed_X = torch.zeros(train_X.size())

                    # update counters
                    self.model["covars_proposed_iter"] = train_X.size()[0]
                    self.model["covars_sampled_iter"] = train_X.size()[0]
                    self.model["response_sampled_iter"] = train_Y.size()[0]

    def __initialize_random_start(
        self, random_start, num_initial_random, random_sampling_method
    ):
        """
        set details for how to randomize at start
        :param random_start (bool): input from user (set during TuneSession.__init__)
        :param num_initial_random (int/None): provided as kwarg input from user (set during TuneSession.__init__)
        :param random_sampling_method (str/None): sampling method for random points. Options: "random" and "latin_hcs"
        (latin hypercube sampling). Provided as kwarg input from user (set during TuneSession.__init__)
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
        SAMPLING_METHODS_LIST = [
            func
            for func in dir(DataSamplers)
            if callable(getattr(DataSamplers, func)) and not func.startswith("__")
        ]
        if (random_sampling_method is not None) & (
            random_sampling_method not in SAMPLING_METHODS_LIST
        ):
            raise Exception(
                "greattunes._initializers.Initializers.__initialize_random_start: The parameter "
                "'random_sampling_method' is not among allowed values ('"
                + "', '".join(SAMPLING_METHODS_LIST)
                + "')."
            )

        # set sampling method
        if random_sampling_method is None:
            self.random_sampling_method = (
                "latin_hcs"  # default to latin hypercube sampling if not specified
            )
        else:
            self.random_sampling_method = random_sampling_method

        # split: train_X, train_Y present
        if (self.train_X is not None) & (self.train_Y is not None):

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
                warnings.warn(
                    "Inconsistent settings for optimization initialization: No initial data provided via "
                    "'train_X' and 'train_Y' but also 'random_start' is set to 'False'. Adjusting to start"
                    " with "
                    + str(self.num_initial_random_points)
                    + " random datapoints."
                )

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
            - self.covars (list of tuples): the user-provided details on covariates. Here using only the length of
            the list, which gives the number of covariates
        :return: num_random (int)
        """

        NUM_COVARS = len(self.covars)
        num_random = int(round(math.sqrt(NUM_COVARS), 0))

        return num_random

    def __initialize_pretty_data(self):
        """
        initialize datasets for covariates and response in pandas dataframe format. This will be the primary format for
        users to interact with the data, and it will keep variables in their original types (integer, continuous or
        categorical).

        contrary to the pretty datasets are the sets 'train_X', 'train_Y' containing the data in the continuous format
        used for all variables in the library backend (incl one-hot encoding of categorical variables)

        the names of the pretty data will be 'x_data' and 'y_data'
        :param:
            - self.train_X (tensor): design matrix of covariates
            - self.train_Y (tensor): corresponding observations
            - self.covar_details (dict of dicts): contains a dict with details about each covariate wuth initial guess
            and range as well as information about which column it is mapped to in train_X dataset used by the Gaussian
            process model behind the scenes and data type of covariate. Includes one-hot encoding for categorical
            variables. For one-hot encoded categorical variables use the naming convention
            <covariate name>_<option name>
        :return x_data (pandas df): named dataframe of covariates. If no data present will be just named, empty df
        :return y_data (pandas df): named dataframe of response. If no data present will be just named, empty df
        """

        # validate that the right attributes are present to proceed
        if not hasattr(self, "covar_details"):
            raise Exception(
                "greattunes._initializers.Initializers.__initialize_pretty_data: attribute "
                "'covar_details' is missing so cannot initialize pretty data. Try running method "
                "'_initializers.Initializers.__initialize_from_covars'."
            )

        # initialize readable versions of train_X, train_Y for user interaction (reading)
        covariate_names = list(self.covar_details.keys())
        x_data = pd.DataFrame(columns=covariate_names)
        y_data = pd.DataFrame(columns=["Response"])

        # add historical data if provided
        if (self.train_X is not None) & (self.train_Y is not None):
            tmp_x = tensor2pretty_covariate(self.train_X, self.covar_details)
            tmp_y = tensor2pretty_response(self.train_Y)

            x_data = x_data.append(tmp_x)
            y_data = y_data.append(tmp_y)

        return x_data, y_data

    def initialize_model(self, model):
        """
        initialize model object for class. Verify that specified model type (given by "model") is supported
        """

        # validate model
        if self._Validators__validate_model_acceptable(model, self.MODEL_LIST):
            model_dict = {
                "model_type": model,
                "model": None,
                "likelihood": None,
                "loglikelihood": None,
                "covars_proposed_iter": 0,
                "covars_sampled_iter": 0,
                "response_sampled_iter": 0,
            }

            return model_dict

    def initialize_acq_func(self, acq_func):
        """
        initialize acq_func object for class. Verify that specified acquisition function type (given by "acq_func") is
        supported
        """

        # validate model
        if self._Validators__validate_model_acceptable(acq_func, self.ACQ_FUNC_LIST):
            acq_func_dict = {
                "type": acq_func,
                "object": None,
            }

            return acq_func_dict
