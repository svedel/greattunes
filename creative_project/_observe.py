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
            "kre8_core.creative_project._observe._print_candidate_to_prompt: provided input 'candidate' is incorrect datatype. Expecting to be of type torch.Tensor")

    # verify that candidate not an empty list
    if not candidate.size()[0] > 0:
        raise Exception("kre8_core.creative_project._observe._print_candidate_to_prompt: provided input 'candidate' is empty. Expecting torch tensor of size 1 X num_covariates")

    # convert 'candidate' to list from tensor (becomes list of lists), and pick the first of these nested lists
    cand_list = candidate.tolist()[0]

    # create string
    input_request = "ITERATION " + str(self.model["covars_proposed_iter"]) + " - NEW datapoint to sample: " + ", ".join(
        [str(x) for x in cand_list])

    return input_request
