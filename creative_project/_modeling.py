import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from creative_project.custom_models.simple_matern_model import SimpleCustomMaternGP


def _set_GP_model(self, **kwargs):
    """
    initiates model object from BoTorch class and defines the associated likelihood function
    ADD NOISE TO OBSERVATIONS?
    :input from class instance
        - self.train_X (torch.tensor): training data for covariates (design matrix)
        - self.train_Y (torch.tensor): responses correcsponding to each observation in design matrix 'train_X'
        - self.model["model"].state_dict() (object): a state dict of the previously trained model, e.g. output of
        model.state_dict()
    :param kwargs:
        - nu (parameter for Matérn kernel under Custom model_type)
    :output: update class instance
        - self.model["model"] (BoTorch model):
        - self.model["loglikelihood"] (BoTorch log-likelihood):
    :return model_retrain_succes_str (str)
    """

    # TODO: check that self.model['model_type'] value is allowed

    # FixedNoiseGP is a BoTorch alternative that also includes a fixed noise estimate on the observations train_Y
    if self.model["model_type"] == "SingleTaskGP":

        # set up the model
        model_obj = SingleTaskGP(self.train_X, self.train_Y)

        # the likelihood
        lh = model_obj.likelihood

        # define the "loss" function
        ll = ExactMarginalLogLikelihood(lh, model_obj)

    # Custom is a custom model based on Matérn kernel
    elif self.model["model_type"] == "Custom":

        nu = kwargs.get("nu")

        # set up the model
        model_obj = SimpleCustomMaternGP(self.train_X, self.train_Y, nu)

        # likelihood
        lh = GaussianLikelihood()

        # define the "loss" function
        ll = ExactMarginalLogLikelihood(lh, model_obj)
    else:
        raise Exception(
            "creative_project._modeling._set_GP_model: unknown 'model_type' ("
            + self.model["model_type"]
            + ") provided. Must be in following list ['Custom', 'SingleTaskGP']"
        )

    # add stored model if present
    if "model" in self.model:
        if self.model["model"] is not None:
            if self.model["model"].state_dict() is not None:

                model_dict = model_obj.state_dict()
                pretrained_dict = self.model["model"].state_dict()

                # filter unnecessary keys
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict
                }

                # overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)

                # load the new state dict
                model_obj.load_state_dict(pretrained_dict)

    # fit the underlying model
    fit_gpytorch_model(ll)

    # return model + likelihood
    self.model["model"] = model_obj
    self.model["likelihood"] = lh
    self.model["loglikelihood"] = ll

    return "Successfully trained GP model"


# kernel transformation mapping
def _GP_kernel_transform(self, x):
    """
    performs transformation of covariates to enable Gaussian Process models to also do Bayesian optimization for integer
    and categorical variables. The transformation is part of the solution described in this paper:
    E.C. Garrido-Merchán and D. Hernandéz-Lobato: Dealing with categorical and integer-valued variables in Bayesian
    Optimization with Gaussian processes, Neurocomputing vol. 380, 7 March 2020, pp. 20-35
    (https://arxiv.org/pdf/1805.03463.pdf, https://www.sciencedirect.com/science/article/abs/pii/S0925231219315619)

    Briefly the transformation applies only to the continuous variables, and is applied only inside the GP kernel. The
     transformation does the following:
    * for integers: integers covariates (still handled via a single continuous covariate in the GP), the continuous GP
    variable is mapped to the nearest integer (via rounding)
    * for categorical: categorical covariates are handled via one-hot encoding to continuous variables in the GP; the
    transformation selects the largest new one-hot encoded variables and assigns it value 1 while setting the values
    of the rest of the one-hot variables to 0.

    :param x (torch tensor, <num_rows> x <num_covariates> (number of GP covariates))
    :param:
        - self.GP_kernel_mapping_covar_identification (list of dicts): contains information about all covariates
        requiring special attention (name, type and which columns in train_X, train_Y)
    :return x_output (torch tensor, <num_rows> x <num_covariates> (number of GP covariates)) with transformation applied
    """

    x_output = x

    for mapped_covar in self.GP_kernel_mapping_covar_identification:

        # case where variable is of type int (integer)
        if mapped_covar["type"] == int:
            x_output[:, mapped_covar["columns"]] = torch.round(
                x[:, mapped_covar["columns"]]
            )

        # case where variable is of type str (categorical)
        elif mapped_covar["type"] == str:

            # identify column of max value
            _, max_index = torch.topk(x[:, mapped_covar["columns"]], 1)

            # set all but column of max value to 0, max value column to 1
            # first set all entries in one-hot variables to 0
            # then run through max one-hot variable in each row to set to 1 (also handles cases of >1 observations)
            x_output[:, mapped_covar["columns"]] = 0.0
            for row_id in range(max_index.size()[0]):
                x_output[
                    row_id, mapped_covar["columns"][max_index[row_id, 0].item()]
                ] = 1.0

    return x_output
