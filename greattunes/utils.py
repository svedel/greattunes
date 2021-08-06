import numpy as np
import pandas as pd
import importlib, inspect, torch
from greattunes.data_format_mappings import (
    pretty2tensor_covariate,
    pretty2tensor_response,
)


def __get_covars_from_kwargs(covars, **kwargs):
    """
    get covariates of observation from "covars" and returns in tensor format. "covars" originates as kwarg in
    _campaign.tell. Validation of content of covars is done in _observe._get_and_verify_covars_input
    :param covars (list or torch tensor of size 1 X num_covars or pandas dataframe):
    :param kwargs if provided covars in pandas format:
        - covar_details (dict of dicts): contains a dict with details about each covariate wuth initial guess and
        range as well as information about which column it is mapped to in train_X dataset used by the Gaussian
        process model behind the scenes and data type of covariate. Includes one-hot encoding for categorical
        variables. For one-hot encoded categorical variables uses the naming convention
        <covariate name>_<option name>
        - covar_mapped_names (list): names of mapped covariates
        - device (torch device)
    :return: covars_candidate_float_tensor (tensor of size 1 X num_covars)
    """

    # verify covars datatype
    if not isinstance(covars, (list, torch.DoubleTensor, pd.DataFrame)):
        raise Exception(
            "greattunes.utils.__get_covars_from_kwargs: datatype of provided 'covars' is not allowed."
            "Only accept types 'list' and 'torch.DoubleTensor', got "
            + str(type(covars))
        )

    # handle case when a list is provided
    if isinstance(covars, list):

        try:
            covars_candidate_float_tensor = torch.tensor([covars], dtype=torch.double)
        except Exception as e:  # to catch any error in reading the provided "covars"
            raise e

    # handle the case where a pandas dataframe is provided
    elif isinstance(covars, pd.DataFrame):

        try:
            covar_details = kwargs.get("covar_details")
            covar_mapped_names = kwargs.get("covar_mapped_names")
            device = kwargs.get("device")

            covars_candidate_float_tensor, _ = pretty2tensor_covariate(
                x_pandas=covars,
                covar_details=covar_details,
                covar_mapped_names=covar_mapped_names,
                device=device,
            )
        except Exception as e:  # to catch any error in reading the provided "covars"
            raise e

    elif isinstance(covars, torch.DoubleTensor):

        # verify that a single-row tensor has been provided
        covars_size_list = list(covars.size())

        # first condition checks for only a single row in covars, second condition checks that there are columns in the
        # row (not just a single-element column vector)
        if covars_size_list[0] == 1 and len(covars_size_list) == 2:
            covars_candidate_float_tensor = covars
        else:
            raise Exception(
                "greattunes.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'."
                " Was expecting torch tensor of size (1,<num_covariates>) but received one of size ("
                + ", ".join([str(ent) for ent in covars_size_list])
                + ")."
            )

    return covars_candidate_float_tensor


def __get_response_from_kwargs(response, **kwargs):
    """
    get response of observation from "response" and returns in tensor format. "response" originates as kwarg in
    _campaign.tell. Validation of response is done in _observe._get_and_verify_response_input
    :param response (list or torch tensor of size 1 X 1 or pandas dataframe):
    :param kwargs:
        - device (torch device)
    :return: response_candidate_float_tensor (tensor of size 1 X 1)
    """

    # verify response datatype
    if not isinstance(response, (list, torch.DoubleTensor, pd.DataFrame)):
        raise Exception(
            "greattunes.utils.__get_response_from_kwargs: datatype of provided 'response' is not allowed."
            "Only accept types 'list' and 'torch.DoubleTensor', got "
            + str(type(response))
        )

    # handle case when a list is provided
    if isinstance(response, list):

        try:
            device = kwargs.get("device")

            response_candidate_float_tensor = torch.tensor(
                [response], dtype=torch.double, device=device
            )
        except Exception as e:  # to catch any error in reading the provided "covars"
            raise e

    # handle the case where a pandas dataframe is provided
    elif isinstance(response, pd.DataFrame):

        try:
            device = kwargs.get("device")
            response_candidate_float_tensor = pretty2tensor_response(
                y_pandas=response, device=device
            )
        except Exception as e:  # to catch any error in reading the provided "covars"
            raise e

    # handle the case where a torch tensor is provided
    elif isinstance(response, torch.DoubleTensor):

        # get size of response
        resp_size_list = list(response.size())

        # check that there's only a single entry in response
        if resp_size_list[0] == 1 and resp_size_list[1] == 1:
            response_candidate_float_tensor = response
        else:
            raise Exception(
                "greattunes.utils.__get_response_from_kwargs: dimension mismatch in provided 'response'."
                " Was expecting torch tensor of size (1,1) but received one of size ("
                + ", ".join([str(ent) for ent in resp_size_list])
                + ")."
            )

    return response_candidate_float_tensor


class DataSamplers:
    """
    class of sample methods (random and structured random) used for initialization and for interdispersed random
    sampling
    """

    @staticmethod
    def random(n_samp, initial_guess, covar_bounds, device):
        """
        randomly samples each covariate within bounds to provice new candidate datapoint
        :param n_samp (int): number of samples to be generated. Defines number of subsegments for each covariate, from
        which each new candidate datapoint are obtained
        :param initial_guess (tensor, 1 X <num covariates>): contains all initial guesses for covariate values
            provided by user
        :param covar_bounds (tensor, 2 X <num covariates>): upper and lower bounds for covariates provided by user
        :param device (torch device): computational device used
        :return: candidates (tensor, n_samp X <num covariates>): tensor of new candidates
        """

        # number of covariates
        # initial_guess has size 1 x <num covariates>
        NUM_COVARS = initial_guess.shape[1]

        # randomly sample each covariate. Use uniform sampling probability within bounds provided (self.covar_bounds).
        # iterate since each covariate has its own bounds on the covariate range.
        # attribute covar_bounds stores lower bounds in row 0, upper bounds in row 1, and first tuple entry in 'size'
        # argument determines number of repeat samples
        candidates_array = np.random.uniform(
            low=covar_bounds[0, :].numpy(),
            high=covar_bounds[1, :].numpy(),
            size=(n_samp, NUM_COVARS),
        )

        # convert to torch tensor. Each row in this tensor is a candidate
        # the chained .double() command converts to array of data type double
        candidates = torch.from_numpy(candidates_array).double().to(device)

        return candidates

    @staticmethod
    def latin_hcs(n_samp, initial_guess, covar_bounds, device):
        """
        structured random sampling of covariates within bounds to provide new candidate. For details on latin hypercube
        sampling please consult https://en.wikipedia.org/wiki/Latin_hypercube_sampling.

        For each covariate, the range of possible outcomes is subdivided in n_samp bins. Each of these bins can only be
        present once across all candidates. Since each covariate is divided into the same number of bins, this is
        equivalent to shuffling the bins for each row independently.

        NOTE: Approach in this method assumes all covariates are continuous. Reconsider when no longer the case

        :param initial_guess (tensor, 1 X <num covariates>): contains all initial guesses for covariate values
            provided by user
        :param covar_bounds (tensor, 2 X <num covariates>): upper and lower bounds for covariates provided by user
        :param n_samp (int): number of samples to be generated. Defines number of subsegments for each covariate, from
        which each new candidate datapoint are obtained
        :param device (torch device): computational device used
        :return: candidates (tensor, n_samp X <num covariates>): tensor of new candidates
        """

        # number of covariates
        NUM_COVARS = initial_guess.shape[
            1
        ]  # initial_guess has size 1 x <num covariates>

        # create array of bins. Each row corresponds to a unique point in the Latin hypercube
        bins = np.zeros((n_samp, NUM_COVARS))
        for i in range(NUM_COVARS):
            bins[:, i] = np.random.permutation(n_samp)

        # sample within each bin for each variable (uniform sampling)
        candidates_array = np.zeros((n_samp, NUM_COVARS))

        for i in range(NUM_COVARS):
            # bin boundaries for this covariate
            # add a final datapoint to get the bin separations (n_samp + 1)
            bin_boundaries_tmp = np.linspace(
                covar_bounds[0, i].item(), covar_bounds[1, i].item(), n_samp + 1
            )

            # random sampling with uniform distribution within bin
            # this has size 1 X n_samp
            candidates_array[:, i] = np.random.uniform(
                low=bin_boundaries_tmp[:-1], high=bin_boundaries_tmp[1:]
            )

        # convert to torch tensor. Each row in this tensor is a candidate
        # the chained .double() command converts to array of data type double
        candidates = torch.from_numpy(candidates_array).double().to(device)

        # return
        return candidates


def classes_from_file(file_path):
    """
    returns names of all classes defined in the file given by 'file_path'
    """

    class_names = []

    for name, cls in inspect.getmembers(
        importlib.import_module(file_path), inspect.isclass
    ):
        if cls.__module__ == file_path:
            class_names.append(name)

    return class_names
