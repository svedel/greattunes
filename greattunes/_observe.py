"""
Methods for observing responses and the associated covariates
"""
import pandas as pd
import numpy as np
import torch
from .utils import __get_covars_from_kwargs, __get_response_from_kwargs
from .data_format_mappings import (
    pretty2tensor_covariate,
    tensor2pretty_covariate,
    tensor2pretty_response,
)


# === Response methods ===
def _get_response_datapoint(self, response):
    """
    gets observation of actual response y. Updates stored data, counters etc.
    NOTE: the new response data counter ("how many responses do we have") is derived from the number of proposed
    covariates, not the number of sampled responses. This in order to allow for a covariate to be reported after the
    response. However, only when .ask-method is rerun will can new covariates and responses be added.
    :depends on:
        - self.model["covars_proposed_iter"] (counter of number of covars proposed)
        - self.train_Y
    :updates:
        - self.train_Y
        - self.model["response_sampled_iter"] (counter of number of responses sampled)
    :param: response (list or torch tensor or None): kwarg input in _campaign.tell (None if not present in tell). This
    provides a programmatic way of providing the response data
    """

    # iteration counter of proposed datapoint
    obs_counter = self.model["covars_proposed_iter"]

    # get and verify response datapoint
    response_datapoint = self._get_and_verify_response_input(response=response)

    # store data
    # first datapoint
    if self.train_Y is None:

        # backend format
        self.train_Y = response_datapoint

        # user-facing pretty format
        self.y_data = tensor2pretty_response(train_Y_sample=response_datapoint)

    # case where sampling this iteration for the second time, overwriting first sampled datapoint
    elif self.train_Y.shape[0] >= obs_counter:

        # backend data format
        # self.train_Y[obs_counter - 1, :] = response_datapoint --- won't work since train_Y has only one element
        self.train_Y[obs_counter - 1] = response_datapoint

        # user-facing pretty format
        # first add replacing row as new row, then remove the to-be-replaced row, and finally reset indicies
        self.y_data = self.y_data.append(
            tensor2pretty_response(train_Y_sample=response_datapoint), ignore_index=True
        )
        self.y_data.drop([obs_counter - 1], inplace=True)
        self.y_data.reset_index(drop=True, inplace=True)

    else:

        # backend data format
        # self.train_Y.append(response_datapoint)
        self.train_Y = torch.cat((self.train_Y, response_datapoint), dim=0)

        # user-facing pretty format
        self.y_data = self.y_data.append(
            tensor2pretty_response(train_Y_sample=response_datapoint)
        )

    # update counter
    self.model["response_sampled_iter"] = obs_counter


def _get_and_verify_response_input(self, **kwargs):
    """
    read and verify response. Assumes only a single input, and does not do any verification.
    :input:
        - self.sampling["method"]: determines how to get the response data (iteratively via input or function
        evaluation). Default model self.sampling["method"] = 'iterative' is set in greattunes.__init__.py
    :kwargs:
        - response (list or torch tensor or None): kwarg input in _campaign.tell (None if not present in tell). This
        provides a programmatic way of providing the response data
    :return response_datapoint (torch Tensor): a single-element tensor containing the returned response datapoint
    """

    kwarg_response = kwargs.get("response")

    # get candidate
    if self.sampling["method"] == "iterative":

        # get from programmatic input
        if kwarg_response is not None:

            # response_datapoint = __get_covars_from_kwargs(kwarg_response)
            response_datapoint = __get_response_from_kwargs(
                kwarg_response, device=self.device
            )

        # get from manual input from prompt
        else:
            additional_text = ""
            response_datapoint = self._read_response_manual_input(additional_text)

    elif self.sampling["method"] == "functions":

        response_datapoint = self._get_response_function_input()

    else:
        raise Exception(
            "greattunes._observe._get_and_verify_response_input: class attribute "
            "self.sampling['method'] has non-permissable value "
            + str(self.sampling["method"])
            + ", must"
            " be in ['iterative', 'functions']."
        )

    if self._Validators__validate_num_response(response_datapoint):
        return response_datapoint
    else:
        raise Exception(
            "greattunes._observe._get_and_verify_response_input: incorrect number of variables provided. Was "
            "expecting input of size (1,1) but received "
            + str(response_datapoint.size())
        )


def _get_response_function_input(self):
    """
    gets response to last entry in self.train_X via response function stored in self.sampling["response_func"].
    Assumes response is univariate.
    :input
        - self.train_X: latest proposed covariates
        - self.sampling["response_func"]: the response function (applied to last row in self.train_X). The function must
        accept a pandas data frame as input (in the format of the pretty data 'x_data' for the class) and must return a
        single element as response; acceptable formats for return are: int, float, any numpy float, any numpy int, list,
        numpy array, pandas dataframe (with response column named "Response").
    :return response_candidate_float_tensor (torch tensor): a 1x1 torch tensor with response to last datapoint
    """

    # the covariates determined as new datapoint by acquisition function
    num_covars = self.train_X.shape[1]
    covars = self.train_X[-1].reshape(1, num_covars)

    # converts covariates for function to pretty format
    covars_pretty = tensor2pretty_covariate(
        train_X_sample=covars, covar_details=self.covar_details
    )

    # the corresponding response
    resp = self.sampling["response_func"](covars_pretty)

    # cast response as tensor
    if isinstance(resp, (int, float, np.floating, np.integer)):
        response_candidate_float_tensor = torch.tensor(
            [[resp]], device=self.device, dtype=self.dtype
        )
    # for types list, numpy array take the last element
    elif type(resp) in {list, np.ndarray}:
        response_candidate_float_tensor = torch.tensor(
            [resp[-1]], device=self.device, dtype=self.dtype
        ).reshape(1, 1)
    # for pandas take the last element from column "Response"
    elif type(resp) == pd.DataFrame:
        response_candidate_float_tensor = torch.tensor(
            [[resp["Response"].iloc[-1]]], device=self.device, dtype=self.dtype
        )
    else:
        raise Exception(
            "greattunes._observe._get_response_function_input: response function provided does not"
            " return acceptable output types ('int','float','list','numpy.ndarray','pandas.DataFrame'), "
            " but returned " + str(type(resp))
        )

    return response_candidate_float_tensor


def _read_response_manual_input(self, additional_text):
    """
    read response provided by user as manual input via prompt
    :param additional_text (str): additional text to display in prompt when sourcing user input;
    to be used in case a datapoint is to be re-sampled
    """

    # read candidate datapoint
    response_candidate_string = input(
        "ITERATION "
        + str(self.model["covars_proposed_iter"])
        + additional_text
        + " - Please provide response: "
    )
    # assumes only a single response provided (i.e. not providing input on multiple parameters and weighing)
    response_candidate_float_tensor = torch.tensor(
        [[float(z) for z in response_candidate_string.split(",")]],
        device=self.device,
        dtype=self.dtype,
    )

    return response_candidate_float_tensor


# === Covariate methods ===
def _print_candidate_to_prompt(self, candidate):
    """
    prints a candidate for new data point to prompt in user-friendly manner.
    :param candidate (torch tensor): one-row tensor of new datapoint to be investigated
    :return input_request (str)
    """

    # verify datatype of candidate
    if not isinstance(candidate, torch.Tensor):
        raise Exception(
            "greattunes.greattunes._observe._print_candidate_to_prompt: provided input 'candidate' is incorrect "
            "datatype. Expecting to be of type torch.Tensor"
        )

    # verify that candidate not an empty list
    if not candidate.size()[0] > 0:
        raise Exception(
            "greattunes.greattunes._observe._print_candidate_to_prompt: provided input 'candidate' is empty. "
            "Expecting torch tensor of size 1 X num_covariates"
        )

    # convert candidate to named tensor
    cand_pretty = tensor2pretty_covariate(
        train_X_sample=candidate, covar_details=self.covar_details
    )

    # add data type to column names
    new_cand_names = [
        i + " (" + str(self.covar_details[i]["type"]) + ")"
        for i in list(cand_pretty.columns)
    ]
    cand_pretty.columns = new_cand_names

    # create string
    input_request = "\tNEW datapoint to sample:\n\t" + cand_pretty.to_string(
        index=False
    ).replace("\n", "\n\t")

    return input_request


def _get_covars_datapoint(self, covars):
    """
    gets observation of actual covars x. Updates stored data, counters etc
    assumes:
        - covars x can only be one iteration ahead of observation y
        - tracking of response observations is managed elsewhere, where it's also ensured that these are assigned to
        right counter
    :param: covars (list or torch tensor or None): kwarg input in _campaign.tell (None if not present in tell). This
    provides a programmatic way of providing the covars data
    """

    # iteration counter of proposed datapoint
    obs_counter = self.model["covars_proposed_iter"]

    # get and verify covars datapoint
    covars_datapoint = self._get_and_verify_covars_input(covars)

    # store data
    # first datapoint
    if self.train_X is None:
        # backend data
        self.train_X = covars_datapoint

        # user-facing pretty data
        self.x_data = tensor2pretty_covariate(
            train_X_sample=covars_datapoint, covar_details=self.covar_details
        )

    # case where sampling this iteration for the second time, overwriting first sampled datapoint
    elif self.train_X.shape[0] >= obs_counter:
        # backend data
        self.train_X[obs_counter - 1, :] = covars_datapoint

        # user-facing data
        # first add replacing row as new row, then remove the to-be-replaced row, and finally reset indicies
        self.x_data = self.x_data.append(
            tensor2pretty_covariate(
                train_X_sample=covars_datapoint, covar_details=self.covar_details
            ),
            ignore_index=True,
        )
        self.x_data.drop([obs_counter - 1], inplace=True)
        self.x_data.reset_index(drop=True, inplace=True)

    else:
        # backend data
        self.train_X = torch.cat((self.train_X, covars_datapoint), dim=0)

        # user-facing pretty data
        self.x_data = self.x_data.append(
            tensor2pretty_covariate(
                train_X_sample=covars_datapoint, covar_details=self.covar_details
            )
        )

    # update counter
    self.model["covars_sampled_iter"] = obs_counter


def _get_and_verify_covars_input(self, covars):
    """
    read and verify covars. Currently focused on manual input only, but to be expanded to allow multiple
    different mechanism (if needed). Allowing up to 'MAX_ITER' repeats of providing the data and updates text to
    prompt to indicate if candidates datapoint not accepted
    :param: covars (list or torch tensor or None): kwarg input in _campaign.tell (None if not present in tell). This
    provides a programmatic way of providing the covars data
    """

    MAX_ITER = 3
    it = 0
    accepted = False

    additional_text = ""

    while not accepted and it < MAX_ITER:

        it += 1

        # read covars
        if covars is not None:
            covars_candidate_float_tensor = __get_covars_from_kwargs(
                covars,
                device=self.device,
                covar_details=self.covar_details,
                covar_mapped_names=self.covar_mapped_names,
            )
        else:
            covars_candidate_float_tensor = self._read_covars_manual_input(
                additional_text
            )

        # verify number of provided elements is correct
        if self._Validators__validate_num_covars(covars_candidate_float_tensor):
            accepted = True
        else:
            additional_text = (
                " (REPEAT, prior attempt had incorrect number of datapoints) "
            )

    if accepted:
        return covars_candidate_float_tensor
    else:
        add_text = ""
        if self.proposed_X is not None:
            add_text = (
                " Was expecting something like '"
                + str(self.proposed_X[-1])
                + "', but got '"
                + str(covars_candidate_float_tensor)
                + "'"
            )
        raise Exception(
            "greattunes._observe._get_and_verify_covars_input: unable to get acceptable covariate input in "
            + str(MAX_ITER)
            + " iterations."
            + add_text
        )


def _read_covars_manual_input(self, additional_text):
    """
    read covars as manual input from prompt
    :param additional_text (str): additional text to be displayed in prompt
    :return covars_candidate_float_tensor (torch.tensor): user-provided covariates
    TODO:
        - make more user friendly in terms of making sure user can make mistakes
            - don't have to remember order of variables
            - can see whether a variable has already been provided
            - will need to accept at the end
    """

    assert isinstance(additional_text, str), (
        "greattunes._observe._read_covars_manual_input: wrong datatype of parameter 'additional_text'. "
        "Was expecting 'str' but received " + str(type(additional_text))
    )

    covars_candidate_string = input(
        "ITERATION "
        + str(self.model["covars_proposed_iter"])
        + additional_text
        + " - Please provide observed covariates separated by commas (covariates: "
        + ", ".join(
            [
                k + " (" + str(self.covar_details[k]["type"]) + ")"
                for k in self.covar_details.keys()
            ]
        )
        + "): "
    )

    # assess correct number of covariates provided
    # convert input str to pandas
    covar_str = covars_candidate_string.split(",")

    if not len(covar_str) == len(self.covar_details):
        raise Exception(
            "greattunes._observe._read_covars_manual_input: incorrect number of covariates ("
            + str(len(covar_str))
            + ") provided, was expecting "
            + str(len(self.covar_details))
        )

    # extract data values, convert to right type, assemble in pretty data format (pandas dataframe)
    covar_converted_types = [None for x in covar_str]  # initialize
    for key in self.covar_details:
        entry = self.covar_details[key]["pandas_column"]
        entry_datatype = self.covar_details[key]["type"]
        if np.isnan(entry_datatype(covar_str[entry])):
            raise Exception(
                "greattunes._observe._read_covars_manual_input: provided '"
                + covar_str[entry]
                + "' becomes 'nan' when converting to expected datatype "
                + str(entry_datatype)
                + " for this "
                "entry position."
            )
        covar_converted_types[entry] = entry_datatype(covar_str[entry])

        # removes trailing/leading empty spaces for string type
        if entry_datatype == str:
            covar_converted_types[entry] = covar_converted_types[entry].strip()

    covar_df_tmp = pd.DataFrame(columns=self.sorted_pandas_columns)
    covar_df_tmp.loc[0] = covar_converted_types

    # convert to tensor using pretty2tensor
    covars_candidate_float_tensor, _ = pretty2tensor_covariate(
        x_pandas=covar_df_tmp,
        covar_details=self.covar_details,
        covar_mapped_names=self.covar_mapped_names,
        device=self.device,
    )

    return covars_candidate_float_tensor
