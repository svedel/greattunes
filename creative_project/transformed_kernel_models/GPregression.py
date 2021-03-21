"""
Apply the transformation by E.C. Garrido-Merchán and D. Hernandéz-Lobato to model types.

Reference:

E.C. Garrido-Merchán and D. Hernandéz-Lobato: Dealing with categorical and integer-valued variables in Bayesian
Optimization with Gaussian processes, Neurocomputing vol. 380, 7 March 2020, pp. 20-35
(https://arxiv.org/pdf/1805.03463.pdf, https://www.sciencedirect.com/science/article/abs/pii/S0925231219315619)
"""

from botorch.models import SingleTaskGP
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from creative_project.transformed_kernel_models.transformation import (
    GP_kernel_transform,
)


class SingleTaskGP_transformed(SingleTaskGP):
    """
    version of SingleTaskGP where input data to kernel model is transformed
    """

    def __init__(
        self, train_X, train_Y, GP_kernel_mapping_covar_identification, likelihood=None,
    ):
        super().__init__(train_X=train_X, train_Y=train_Y)
        self.GP_kernel_mapping_covar_identification = (
            GP_kernel_mapping_covar_identification
        )

        # self.covar_module = ScaleKernel(
        #     MaternKernel(
        #         nu=2.5,
        #         ard_num_dims=train_X.shape[-1],
        #         batch_shape=self._aug_batch_shape,
        #         lengthscale_prior=GammaPrior(3.0, 6.0),
        #     ),
        #     batch_shape=self._aug_batch_shape,
        #     outputscale_prior=GammaPrior(2.0, 0.15),
        # )

    def forward(self, x):
        # x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(
            GP_kernel_transform(x, self.GP_kernel_mapping_covar_identification)
        )
        return MultivariateNormal(mean_x, covar_x)
