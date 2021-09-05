"""
Apply the transformation by E.C. Garrido-Merchán and D. Hernandéz-Lobato to model types.

Reference:

E.C. Garrido-Merchán and D. Hernandéz-Lobato: Dealing with categorical and integer-valued variables in Bayesian
Optimization with Gaussian processes, Neurocomputing vol. 380, 7 March 2020, pp. 20-35
(https://arxiv.org/pdf/1805.03463.pdf, https://www.sciencedirect.com/science/article/abs/pii/S0925231219315619)
"""

from botorch.models import FixedNoiseGP, HeteroskedasticSingleTaskGP, SingleTaskGP
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from greattunes.transformed_kernel_models.transformation import GP_kernel_transform


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

    def forward(self, x):
        # x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(
            GP_kernel_transform(x, self.GP_kernel_mapping_covar_identification)
        )
        return MultivariateNormal(mean_x, covar_x)


class FixedNoiseGP_transformed(FixedNoiseGP):
    """
    version of FixedNoiseGP which scales input data to kernel model with transform
    """

    def __init__(
        self,
        train_X,
        train_Y,
        train_Yvar,
        GP_kernel_mapping_covar_identification,
        likelihood=None,
    ):
        super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self.GP_kernel_mapping_covar_identification = (
            GP_kernel_mapping_covar_identification
        )

    def forward(self, x):
        # x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(
            GP_kernel_transform(x, self.GP_kernel_mapping_covar_identification)
        )
        return MultivariateNormal(mean_x, covar_x)


class HeteroskedasticSingleTaskGP_transformed(HeteroskedasticSingleTaskGP):
    """
    version of HeteroskedasticSingleTaskGP which scales input data to kernel model with transform
    """

    def __init__(
        self,
        train_X,
        train_Y,
        train_Yvar,
        GP_kernel_mapping_covar_identification,
        likelihood=None,
    ):
        super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self.GP_kernel_mapping_covar_identification = (
            GP_kernel_mapping_covar_identification
        )

    def forward(self, x):
        # x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(
            GP_kernel_transform(x, self.GP_kernel_mapping_covar_identification)
        )
        return MultivariateNormal(mean_x, covar_x)
