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

    return (
        "ITERATION  "
        + str(self.model["response_sampled_iter"])
        + " - Successfully retrained GP model"
    )
