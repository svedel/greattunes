import numpy as np
import torch


def __get_covars_from_kwargs(covars):
    """
    get covariates of observation from "covars" and returns in tensor format. "covars" originates as kwarg in
    _campaign.tell. Validation of content of covars is done in _observe._get_and_verify_covars_input
    :param covars (list or torch tensor of size 1 X num_covars):
    :return: covars_candidate_float_tensor (tensor of size 1 X num_covars)
    """

    # verify covars datatype
    if not isinstance(covars, (list, torch.DoubleTensor)):
        raise Exception(
            "creative_project.utils.__get_covars_from_kwargs: datatype of provided 'covars' is not allowed."
            "Only accept types 'list' and 'torch.DoubleTensor', got "
            + str(type(covars))
        )

    # handle case when a list is provided
    if isinstance(covars, list):

        try:
            covars_candidate_float_tensor = torch.tensor([covars], dtype=torch.double)
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
                "creative_project.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'."
                " Was expecting torch tensor of size (1,<num_covariates>) but received one of size "
                + str(covars_size_list)
            )

    return covars_candidate_float_tensor


class DataSamplers:
    """
    class of sample methods (random and structured random) used for initialization and for interdispersed random
    sampling
    """

    @staticmethod
    def random(initial_guess, covar_bounds, device, dtype):
        """
        randomly samples each covariate within bounds to provice new candidate datapoint
        :param initial_guess (tensor, 1 X <num covariates>): contains all initial guesses for covariate values
            provided by user
        :param covar_bounds (tensor, 2 X <num covariates>): upper and lower bounds for covariates provided by user
        :param device (torch device): computational device used
        :param dtype (torch dtype): tensor data type of output
        :return: candidate (tensor, 1 X <num covariates>): new candidate
        """

        # number of covariates
        NUM_COVARS = initial_guess.shape[1]  # initial_guess has size 1 x <num covariates>

        # initialize
        candidate = torch.empty((1, NUM_COVARS), dtype=dtype, device=device)

        # randomly sample each covariate. Use uniform sampling probability within bounds provided (self.covar_bounds).
        # iterate since each covariate has its own bounds on the covariate range.
        # attribute covar_bounds stores lower bounds in row 0, upper bounds in row 1, and torch.rand samples uniformly
        for i in range(NUM_COVARS):
            candidate[0, i] = (covar_bounds[1, i] - covar_bounds[0, i]) * torch.rand(1) \
                              + covar_bounds[0, i]

        return candidate


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
        NUM_COVARS = initial_guess.shape[1]  # initial_guess has size 1 x <num covariates>

        # create array of bins. Each row corresponds to a unique point in the Latin hypercube
        bins = np.zeros((n_samp, NUM_COVARS))
        for i in range(NUM_COVARS):
            bins[:, i] = np.random.permutation(n_samp)

        # sample within each bin for each variable (uniform sampling)
        candidates_array = np.zeros((n_samp, NUM_COVARS))

        for i in range(NUM_COVARS):
            # bin boundaries for this covariate
            # add a final datapoint to get the bin separations (n_samp + 1)
            bin_boundaries_tmp = np.linspace(covar_bounds[0, i].item(), covar_bounds[1, i].item(), n_samp+1)

            # random sampling with uniform distribution within bin
            # this has size 1 X n_samp
            candidates_array[:, i] = np.random.uniform(low=bin_boundaries_tmp[:-1], high=bin_boundaries_tmp[1:])

        # convert to torch tensor. Each row in this tensor is a candidate
        # the chained .double() command converts to array of data type double
        candidates = torch.from_numpy(candidates_array).double().to(device)

        # return
        return candidates
