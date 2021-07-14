from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

from greattunes.transformed_kernel_models.transformation import GP_kernel_transform


class SimpleCustomMaternGP(ExactGP, GPyTorchModel):
    """
    Simple customer Gaussian Process model with Matérn kernel and Gaussian likelihood model
    """

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, nu, GP_kernel_mapping_covar_identification):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.GP_kernel_mapping_covar_identification = (
            GP_kernel_mapping_covar_identification
        )
        self.mean_module = ConstantMean()

        if nu is not None:
            self.covar_module = ScaleKernel(
                base_kernel=MaternKernel(
                    nu=nu, ard_num_dims=train_X.shape[-1]
                ),  # set parameter nu in Matern kernel
            )
        else:
            self.covar_module = ScaleKernel(
                base_kernel=MaternKernel(ard_num_dims=train_X.shape[-1]),
                # parameter nu in Matern kernel defauls to 2.5
            )
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(
            GP_kernel_transform(x, self.GP_kernel_mapping_covar_identification)
        )
        # covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
