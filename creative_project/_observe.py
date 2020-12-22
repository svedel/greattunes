"""
Methods for observing responses and the associated covariates
"""
import torch


def _get_and_verify_response_input(self):
    """
    read and verify response. Assumes only a single input, and does not do any verification.
    :input:
        - self.sampling["method"]: determines how to get the response data (manual input or function evaluation)
    :return response_datapoint (torch Tensor): a single-element tensor containing the returned response datapoint
    """

    # get candidate
    if self.sampling["method"] == "manual":

        additional_text = ""

        response_datapoint = self._read_response_manual_input(additional_text)

    elif self.sampling["method"] == "functions":

        response_datapoint = self._get_response_function_input()

    else:
        raise Exception(
            "creative_project._observe._get_and_verify_response_input: class attribute "
            "self.sampling['method'] has non-permissable value "
            + str(self.sampling["method"])
            + ", must"
            " be in ['manual', 'functions']."
        )

    return response_datapoint


def _get_response_function_input(self):
    """
    gets response to last entry in self.train_X via response function stored in self.sampling["response_func"].
    Assumes response is univariate.
    :input
        - self.train_X: latest proposed covariates
        - self.sampling["response_func"]: the response function (applied to last row in self.train_X)
    :return response_candidate_float_tensor (torch tensor): a 1x1 torch tensor with response to last datapoint
    """

    # the covariates determined as new datapoint by acquisition function
    covars = self.train_X[-1]

    # the corresponding response
    resp = self.sampling["response_func"](covars)
    response_candidate_float_tensor = torch.tensor(
        [[resp]], device=self.device, dtype=self.dtype
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


def _print_candidate_to_prompt(self, candidate):
    """
    prints a candidate for new data point to prompt in user-friendly manner.
    :param candidate (torch tensor): one-row tensor of new datapoint to be investigated
    :return input_request (str)
    """

    # verify datatype of candidate
    if not isinstance(candidate, torch.Tensor):
        raise Exception(
            "kre8_core.creative_project._observe._print_candidate_to_prompt: provided input 'candidate' is incorrect "
            "datatype. Expecting to be of type torch.Tensor"
        )

    # verify that candidate not an empty list
    if not candidate.size()[0] > 0:
        raise Exception(
            "kre8_core.creative_project._observe._print_candidate_to_prompt: provided input 'candidate' is empty. "
            "Expecting torch tensor of size 1 X num_covariates"
        )

    # convert 'candidate' to list from tensor (becomes list of lists), and pick the first of these nested lists
    cand_list = candidate.tolist()[0]

    # create string
    input_request = (
        "ITERATION "
        + str(self.model["covars_proposed_iter"])
        + " - NEW datapoint to sample: "
        + ", ".join([str(x) for x in cand_list])
    )

    return input_request


def _covars_datapoint_observation(self):
    """
    gets observation of actual covars x. Updates stored data, counters etc
    assumes:
        - covars x can only be one iteration ahead of observation y
        - tracking of response observations is managed elsewhere, where it's also ensured that these are assigned to
        right counter
    """

    # iteration counter of proposed datapoint
    obs_counter = self.model["covars_proposed_iter"]

    # get and verify covars datapoint
    covars_datapoint = self._get_and_verify_covars_input()

    # store data
    # first datapoint
    if self.train_X is None:
        self.train_X = covars_datapoint

    # case where sampling this iteration for the second time, overwriting first sampled datapoint
    elif self.train_X.shape[0] >= obs_counter:
        self.train_X[obs_counter - 1, :] = covars_datapoint
    else:
        self.train_X = torch.cat((self.train_X, covars_datapoint), dim=0)

    # update counter
    self.model["covars_sampled_iter"] = obs_counter


def _get_and_verify_covars_input(self):
    """
    read and verify covars. Currently focused on manual input only, but to be expanded to allow multiple
    different mechanism (if needed). Allowing up to 'MAX_ITER' repeats of providing the data and updates text to
    prompt to indicate if candidates datapoint not accepted
    """

    MAX_ITER = 3
    it = 0
    accepted = False

    additional_text = ""

    while not accepted and it < MAX_ITER:

        it += 1

        # read covars
        covars_candidate_float_tensor = self._read_covars_manual_input(additional_text)

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
            "creative_project._observe._get_and_verify_covars_input: unable to get acceptable covariate input in "
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
        "creative_project._observe._read_covars_manual_input: wrong datatype of parameter 'additional_text'. "
        "Was expecting 'str' but received " + str(type(additional_text))
    )

    # read candidate
    covars_candidate_string = input(
        "ITERATION "
        + str(self.model["covars_proposed_iter"])
        + additional_text
        + " - Please provide observed covariates separated by commas (proposed datapoint: "
        + ", ".join([str(tmp.item()) for tmp in self.proposed_X[-1]])
        + " covariates): "
    )
    covars_candidate_float_tensor = torch.tensor(
        [[float(z) for z in covars_candidate_string.split(",")]],
        device=self.device,
        dtype=self.dtype,
    )

    return covars_candidate_float_tensor
