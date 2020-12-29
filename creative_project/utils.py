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
        raise Exception("creative_project.utils.__get_covars_from_kwargs: datatype of provided 'covars' is not allowed."
                        "Only accept types 'list' and 'torch.DoubleTensor', got " + str(type(covars)))

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
        if (covars_size_list[0] == 1 and len(covars_size_list) == 2):
            covars_candidate_float_tensor = covars
        else:
            raise Exception("creative_project.utils.__get_covars_from_kwargs: dimension mismatch in provided 'covars'."
                            " Was expecting torch tensor of size (1,<num_covariates>) but received one of size "
                            + str(covars_size_list)
                            )

    return covars_candidate_float_tensor






