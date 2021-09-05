import torch
import warnings
from ._initializers import Initializers
from ._acq_func import AcqFunction
from ._version import __version__


class TuneSession(Initializers, AcqFunction):
    """
    user-facing functionality to balance exploration and exploitation for creative projects.
    Note: Initializers is a child of Validators class (from ._validators.py) so all validator methods are available to
    TuneSession
    """

    # Initialize class instance
    def __init__(
        self,
        covars,
        model="SingleTaskGP",
        acq_func="ExpectedImprovement",
        random_start=True,
        random_step_cadence=10,
        **kwargs,
    ):
        """
        defines the optimization model. Uses BoTorch underneath the hood, so any BoTorch model and aqcuisition function
        can be used. "model" and "aqc_func" must follow BoTorch nomeclature. A list of custom models is also
        available/under development to be included.

        :param covars (list of tuples): each entry (tuple) must contain (<initial_guess>, <min>, <max>) for each input
        variable. Currently only allows for continuous variables. Aspiration: Data type of input will be preserved and
        code be adapted to accommodate both integer and categorical variables.
        :param model str: sets surrogate model (currently allow "SingleTaskGP" and "SimpleCustomMaternGP")
        :param acq_func (str): sets acquisition function (list of available acquisition functions available as
        self.ACQ_FUNC_LIST)
        :param random_start (bool): determines whether to start from random. If set to True and previously obtained
        data provided via train_X and train_Y, there will still be random samples at start
        :param random_step_cadence (int/None): proposes a random datapoint (not determined by Bayesian optimization)
        every 'random_step_cadence' iterations. Runs no random datapoints if set to None. This helps in finding the
        global optimum.
        :kwargs
            - beta (float): tradeoff parameter for acquisition functions UpperConfidenceBound, qUpperConfidenceBound
            - kernel (str): kernel used to define custom GP models (not currently in use)
            - nu (float): kernel parameter for Matern kernel
            - num_fantasies (int): number of realizations for generating estimates for acquisition functions
            qKnowledgeGradient, NoisyExpectedImprovement
            - num_initial_random (int): number of initial random points. Only changes anything if 'random_start' is
            True
            - random_sampling_method (str): sampling method for random points. Options: "random" and "latin_hcs" (latin
            hypercube sampling)
            - sampler (BoTorch acqusition sampler object): sampler for picking datapoints for series of acquisition
            functions. Details on available samplers on https://botorch.org/api/sampling.html.
            - train_X (torch.tensor of dtype=torch.double): design matrix of covariates (batch_shape X num_obs X
            num_training_features OR num_obs X num_training_features)
            - train_Y (torch.tensor of dtype=torch.double): observations (batch_shape X num_obs X num_output_models
            [allows for batched models] OR num_obs X num_output_models)
            - train_Yvar (torch.tensor of dtype=torch.double): observations of noise in the same shape as 'train_Y'.
            Used by some models which include observation noise (FixedNoiseGP, HeteroskedacticSingleTaskGP)
        """

        # === Computational settings ===
        # determines computation device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # data type for tensors
        self.dtype = torch.double

        # === Parameter initialization ===

        # define kernel --- THIS SHOULD BE DONE MORE ELEGANTLY. If not present as input, these will be set to None
        self.nu = kwargs.get("nu")  # get value of nu for Matern kernel

        # define acquisition function parameters
        # number of realizations to average in qKnowledgeGradient, NoisyExpectedImprovement
        # if no input provided and models requiring it are picked, select a number for it
        self.num_fantasies = kwargs.get("num_fantasies")
        if (
            acq_func in ["qKnowledgeGradient", "NoisyExpectedImprovement"]
            and self.num_fantasies is None
        ):
            self.num_fantasies = 20
            if acq_func == "qKnowledgeGradient":
                self.num_fantasies = 64

        # tradeoff parameter for UpperConfidenceBound, qUpperConfidenceBound
        self.beta = kwargs.get("beta")
        if (
            acq_func in ["UpperConfidenceBound", "qUpperConfidenceBound"]
            and self.beta is None
        ):
            self.beta = 0.2

        # === Model and optimization initialization ===
        # define list of all models
        from ._modeling import _models_list

        self.MODEL_LIST = _models_list()

        # define the model
        self.model = self.initialize_model(model=model)

        # list available acquisition functions
        AcqFunction.__init__(
            self
        )  # creates ACQ_FUNC_LIST attribute by running constructor of AcqFunc parent class

        # define acquisition function
        self.acq_func = self.initialize_acq_func(acq_func=acq_func)

        # define sampling functions. initialize as iterative, which means using ask-tell (either manual or automatic).
        # Will be updated if method "auto" is used (use when sampling function known)
        sampling_type = "iterative"
        self.sampling = {"method": sampling_type, "response_func": None}

        # sampler for picking data points
        # if none provided but one is needed, pick a SobolQMCNormal sampler from BoTorch
        self.sampler = kwargs.get("sampler")
        list_acqf_sampler = [
            "qExpectedImprovement",
            "qNoisyExpectedImprovement",
            "qProbabilityOfImprovement",
            "qSimpleRegret",
            "qUpperConfidenceBound",
        ]
        if acq_func in list_acqf_sampler and self.sampler is None:
            from botorch.sampling import SobolQMCNormalSampler

            self.sampler = SobolQMCNormalSampler(1024)

        # === data for training ===
        # initialize the data (initial guess) and bounds on covars
        (
            self.initial_guess,
            self.covar_bounds,
        ) = self._Initializers__initialize_from_covars(covars)
        self.covars = covars  # store provided 'covars' as hidden attribute

        # initialize data for training and storage
        #     - self.train_X (num_covars X num_obs): observed design matrix
        #     - self.train_Y (1 X num_obs): observed response
        #     - self.proposed_X (num_covars X num_obs): matrix of proposed new covars datapoints to sample at each
        #     iteration
        # grab training data if passed through kwargs. If any is not present 'train_X' and/or 'train_Y' will be set to
        # None
        self._Initializers__initialize_training_data(
            train_X=kwargs.get("train_X"), train_Y=kwargs.get("train_Y")
        )

        # user-provided noise levels (if needed)
        # if no 'train_Yvar' is provided but the selected model needs it, then it will throw an error. That is expected
        # behavior
        self.train_Yvar = kwargs.get("train_Yvar")

        # === initialize pretty data (for user interaction) ===
        self.x_data, self.y_data = self._Initializers__initialize_pretty_data()

        # === Random start settings ===
        # set plan for initialization with random samples. In some cases (if train_X, train_Y is accepted) will not set
        # any random initial points
        self._Initializers__initialize_random_start(
            random_start=random_start,
            num_initial_random=kwargs.get("num_initial_random"),
            random_sampling_method=kwargs.get("random_sampling_method"),
        )

        # === random iterations to improve convergence ===
        # set cadence for random iterations, i.e. iterations in which the candidate datapoint proposed is obtained
        # from random sampling instead of from Bayesian optimization. A randomly generated datapoint is taken every
        # 'random_step' iterations.
        # Two special cases to consider
        #   - If set to None, no iterations with randomly generated datapoints are used (after
        #   any random initialization)
        #   - If set to 1 there would be only randomly sampled points. A warning is issued
        if random_step_cadence == 1:
            warnings.warn(
                "The choice of 'random_step_cadence' means that no proposed datapoints will be generated via"
                "Bayesian optimization, i.e. that that there will be only random sampling and no optimization"
            )
        self.random_step_cadence = random_step_cadence

        # === initialize best candidate ===
        # best observed candidate (best response) [self.best_response_value 1 X num_obs tensor], together with
        # corresponding covariates [self.covariates_best_response_value num_covars X num_obs tensor]
        self._Initializers__initialize_best_response()

    def __repr__(self):
        """
        define representation of the instantiated class (what's returned if instantiated class is executed [not any
        methods, just the class])
        :return: str
        """
        deep_str = f"covars={self.covars!r}, model={self.model['model_type']!r}, acq_func={self.acq_func['type']!r}"

        if self.train_X is not None:
            deep_str += f", train_X={self.train_X!r}"

        if self.train_Y is not None:
            deep_str += f", train_Y={self.train_Y!r}"

        if self.nu is not None:
            deep_str += f", nu={self.nu!r}"

        return f"TuneSession(" + deep_str + f")"

    def __str__(self):
        """
        define str of the instantiated class (defines what's returned to command print(<instantiated_class>)
        :return: str
        """
        deep_str = f"covars={self.covars}, model={self.model['model_type']!r}, acq_func={self.acq_func['type']!r}"

        if self.train_X is not None:
            deep_str += f", train_X={self.train_X}"

        if self.train_Y is not None:
            deep_str += f", train_Y={self.train_Y}"

        if self.nu is not None:
            deep_str += f", nu={self.nu}"

        return f"TuneSession(" + deep_str + f")"

    # import methods
    from ._campaign import auto, ask, tell
    from ._observe import (
        _get_response_datapoint,
        _get_and_verify_response_input,
        _get_response_function_input,
        _read_response_manual_input,
        _get_covars_datapoint,
        _get_and_verify_covars_input,
        _read_covars_manual_input,
        _print_candidate_to_prompt,
    )
    from ._modeling import _set_GP_model, _mapped_noise_from_model
    from ._best_response import (
        _find_max_response_value,
        _update_max_response_value,
        current_best,
        _update_proposed_data,
        best_predicted,
        _evaluate_model,
        _find_best_predicted,
    )
    from ._plot import (
        _covars_ref_plot_1d,
        plot_1d_latest,
        predictive_results,
        plot_convergence,
        plot_best_objective,
    )
