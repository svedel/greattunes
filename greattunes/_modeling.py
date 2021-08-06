from botorch.fit import fit_gpytorch_model
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from greattunes.custom_models.simple_matern_model import SimpleCustomMaternGP
from greattunes.transformed_kernel_models.GPregression import (
    SingleTaskGP_transformed,
    FixedNoiseGP_transformed,
)
from .utils import classes_from_file


def _models_list():
    """
    returns a list of all available model types
    """

    # get all transformed botorch models
    transf_models = classes_from_file(
        "greattunes.transformed_kernel_models.GPregression"
    )
    transf_models = [
        x.split("_")[0] for x in transf_models
    ]  # remove "_transformed" appendix on names

    # get all custom models
    # TODO: pick up any file in greattunes.custom_models and extract all classes define in it
    cust_models = classes_from_file("greattunes.custom_models.simple_matern_model")

    model_list = transf_models + cust_models

    return model_list


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

    # FixedNoiseGP is a BoTorch alternative that also includes a fixed noise estimate on the observations train_Y
    if self.model["model_type"] == "SingleTaskGP":

        # set up the model
        model_obj = SingleTaskGP_transformed(
            self.train_X, self.train_Y, self.GP_kernel_mapping_covar_identification
        )

        # the likelihood
        lh = model_obj.likelihood

        # define the "loss" function
        ll = ExactMarginalLogLikelihood(lh, model_obj)

    # Custom is a custom model based on Matérn kernel
    elif self.model["model_type"] == "SimpleCustomMaternGP":

        nu = kwargs.get("nu")

        # set up the model
        model_obj = SimpleCustomMaternGP(
            self.train_X, self.train_Y, nu, self.GP_kernel_mapping_covar_identification
        )

        # likelihood
        lh = GaussianLikelihood()

        # define the "loss" function
        ll = ExactMarginalLogLikelihood(lh, model_obj)
    else:
        raise Exception(
            "greattunes._modeling._set_GP_model: unknown 'model_type' ("
            + self.model["model_type"]
            + ") provided. Must be in following list ['Custom', 'SingleTaskGP']"
        )

    # add stored model if present
    if "model" in self.model:
        if self.model["model"] is not None:
            if self.model["model_type"] != "SimpleCustomMaternGP":
                if self.model["model"].state_dict() is not None:

                    model_dict = model_obj.state_dict()
                    pretrained_dict = self.model["model"].state_dict()

                    # filter unnecessary keys
                    pretrained_dict = {
                        k: v for k, v in pretrained_dict.items() if k in model_dict
                    }

                    # overwrite entries in the existing state dict
                    model_dict.update(pretrained_dict)

                    # Load parameters without standard shape checking.
                    # model_obj.load_strict_shapes(False)

                    # load the new state dict
                    model_obj.load_state_dict(pretrained_dict)

    # fit the underlying model
    fit_gpytorch_model(ll)

    # return model + likelihood
    self.model["model"] = model_obj
    self.model["likelihood"] = lh
    self.model["loglikelihood"] = ll

    return "Successfully trained GP model"
