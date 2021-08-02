import torch
from botorch.acquisition import (
    ExpectedImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    qExpectedImprovement,
    qKnowledgeGradient,
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
    UpperConfidenceBound,
)
from botorch.optim import optimize_acqf

from greattunes.utils import DataSamplers


class AcqFunction:
    """
    all functionality for acquisition functions
    """

    def __init__(self):
        # list of available acquisition functions
        # this is kept as a separate method because all acquisition functions are loaded during method execution, but
        # those should not be loaded as default each time AcqFunction is initialized
        self.ACQ_FUNC_LIST = self.acq_funcs_list()

    def acq_funcs_list(self):
        """
        dynamically create a list of all available acquisition functions for the framework
        """

        from botorch import acquisition
        from inspect import isclass, getmembers

        IGNORE_LIST = [
            "AcquisitionFunction",
            "AnalyticAcquisitionFunction",
            "ConstrainedMCObjective",
            "MultiObjectiveMCAcquisitionFunction",
            "MultiObjectiveAnalyticAcquisitionFunction",
            "MCAcquisitionFunction",
            "MCAcquisitionObjective",
            "CostAwareUtility",
            "GenericCostAwareUtility",
            "InverseCostWeightedUtility",
            "FixedFeatureAcquisitionFunction",
            "GenericMCObjective",
            "IdentityMCObjective",
            "LinearMCObjective",
            "OneShotAcquisitionFunction",
            "ScalarizedObjective",
        ]

        ACQ_FUNC_LIST = [
            cl[0] for cl in getmembers(acquisition, isclass) if cl[0] not in IGNORE_LIST
        ]

        return ACQ_FUNC_LIST

    def set_acq_func(self):
        """
        set the acquisition function
        TODO:
            - assert that model has been initiated (also requires training data)
        """

        if self.model["model"] is None:
            raise Exception(
                "greattunes.greattunes._acq_func.AcqFunction.set_acq_func: no surrogate model set "
                "(self.model['model'] is None)"
            )
        if self.train_Y is None:
            raise Exception(
                "greattunes.greattunes._acq_func.AcqFunction.set_acq_func: no training data provided "
                "(self.train_Y is None)"
            )

        if not self.acq_func["type"] in self.ACQ_FUNC_LIST:
            raise Exception(
                "greattunes.greattunes._acq_func.AcqFunction.set_acq_func: unsupported acquisition function "
                "name provided. '"
                + self.acq_func["type"]
                + "' not in list of supported acquisition functions ["
                + ", ".join(self.ACQ_FUNC_LIST)
                + "]."
            )

        if self.acq_func["type"] == "ExpectedImprovement":
            self.acq_func["object"] = ExpectedImprovement(
                model=self.model["model"], best_f=self.train_Y.max().item()
            )

        elif self.acq_func["type"] == "NoisyExpectedImprovement":

            # get number of realizations (NUM_FANTASIES) from attribute
            NUM_FANTASIES = self.num_fantasies

            # set acquisition function
            self.acq_func["object"] = NoisyExpectedImprovement(
                model=self.model["model"],
                X_observed=self.train_X,
                num_fantasies=NUM_FANTASIES,
            )
        elif self.acq_func["type"] == "PosteriorMean":
            self.acq_func["object"] = PosteriorMean(model=self.model["model"])

        elif self.acq_func["type"] == "ProbabilityOfImprovement":
            self.acq_func["object"] = ProbabilityOfImprovement(
                model=self.model["model"], best_f=self.train_Y.max().item()
            )

        elif self.acq_func["type"] == "qExpectedImprovement":
            sampler = self.sampler
            self.acq_func["object"] = qExpectedImprovement(
                model=self.model["model"],
                best_f=self.train_Y.max().item(),
                sampler=sampler,
            )

        elif self.acq_func["type"] == "qNoisyExpectedImprovement":
            sampler = self.sampler
            self.acq_func["object"] = qNoisyExpectedImprovement(
                model=self.model["model"], X_baseline=self.train_X, sampler=sampler
            )

        elif self.acq_func["type"] == "qProbabilityOfImprovement":
            sampler = self.sampler
            self.acq_func["object"] = qProbabilityOfImprovement(
                model=self.model["model"],
                best_f=self.train_Y.max().item(),
                sampler=sampler,
            )
        elif self.acq_func["type"] == "qSimpleRegret":
            sampler = self.sampler
            self.acq_func["object"] = qSimpleRegret(
                model=self.model["model"], sampler=sampler
            )

        elif self.acq_func["type"] == "qUpperConfidenceBound":
            BETA = self.beta
            sampler = self.sampler
            self.acq_func["object"] = qUpperConfidenceBound(
                model=self.model["model"], beta=BETA, sampler=sampler
            )

        elif self.acq_func["type"] == "qKnowledgeGradient":

            # get number of realizations (NUM_FANTASIES) from attribute
            NUM_FANTASIES = self.num_fantasies

            # set the acquisition function model
            self.acq_func["object"] = qKnowledgeGradient(
                model=self.model["model"], num_fantasies=NUM_FANTASIES
            )

        elif self.acq_func["type"] == "qMaxValueEntropy":
            # generate candidates randomly
            candidate_set = torch.rand(1000, self.covar_bounds.size(1))
            candidate_set = (
                self.covar_bounds[0]
                + (self.covar_bounds[1] - self.covar_bounds[0]) * candidate_set
            )
            self.acq_func["object"] = qMaxValueEntropy(
                model=self.model["model"], candidate_set=candidate_set
            )

        elif self.acq_func["type"] == "qMultiFidelityMaxValueEntropy":
            # generate candidates randomly
            candidate_set = torch.rand(1000, self.covar_bounds.size(1))
            candidate_set = (
                self.covar_bounds[0]
                + (self.covar_bounds[1] - self.covar_bounds[0]) * candidate_set
            )
            self.acq_func["object"] = qMultiFidelityMaxValueEntropy(
                model=self.model["model"], candidate_set=candidate_set
            )

        elif self.acq_func["type"] == "UpperConfidenceBound":
            BETA = self.beta
            self.acq_func["object"] = UpperConfidenceBound(
                model=self.model["model"], beta=BETA
            )

        else:
            raise Exception(
                "greattunes.greattunes._acq_func.AcqFunction.set_acq_func: acquisition function '"
                + self.acq_func["type"]
                + "' has not yet been implemented."
            )

    def __initialize_acq_func(self):
        """
        initialize acquistion function if any response data present
        :input:
            - self.train_Y (no action if this is None)
        :output:
            - self.acq_func["object"] (update with acquisition function object)
        """

        if self.train_Y is not None:
            self.set_acq_func()

    def identify_new_candidate(self):
        """
        identify next candidate covars datapoint. Special case for first iteration: in case no training
        data provided, will start from initial guess provided via "covars" during class instance
        initialization.
        assumes:
            - class instance initialized (specifically have run methods __initialize_from_covars(covars),
            __initialize_training_data())
            - acquisition function self.acq_func["object"] has been initialized via method __initialize_acq_func if
            response data is stored (more specifically self.model["covars_sampled_iter"] > 0).
        :input:
            - self.start_from_guess (bool)
            - self.model["covars_sampled_iter"] (int)
            - self.initial_guess
            - self.acq_func["object"]
            - self.covars_bounds
        :return candidate (1 x num_covars tensor)
        """

        # determine whether to use random or bayesian sampling based on iteration number
        sample_type = self.__random_or_bayesian()

        # select random or bayesian candidate
        if sample_type == "random":
            candidate = self.random_candidate()

        elif sample_type == "bayesian":

            # optimize acquisition function
            BATCH_SIZE = 1

            candidate, _ = optimize_acqf(
                acq_function=self.acq_func["object"],
                bounds=self.covar_bounds,
                q=BATCH_SIZE,
                num_restarts=10,
                raw_samples=512,  # used for intialization heuristic
                options={"maxiter": 200},
            )

        return candidate

    def __random_or_bayesian(self):
        """
        determine whether next candidate datapoint should be determined by random sampling or Bayesian optimization
        :param
            - self.model["covars_sampled_iter"] (int): number of covariate datapoints sampled so far
            - self.num_initial_random_points (int): number of initial random datapoints (data points obtained from
            random sampling and not from the acquisition function). If set to 0, there's no random start
            - self.random_step_cadence (int/None): cadence of when a randomly sampled datapoint is used in between
            points obtained from acquisition function
        :return: sample_type (str): "random" or "bayesian"
        """

        # initialize output
        sample_type = "bayesian"

        # self.model["covars_sampled_iter"] counts the already sampled iterations, while the present method is
        # determining sampling type of the NEXT iteration (self.model["covars_sampled_iter"] + 1). Defining local
        # variable to keep track of iteration number for which sampling type is being determined
        current_iteration = self.model["covars_sampled_iter"] + 1

        # random initialization
        if current_iteration <= self.num_initial_random_points:
            sample_type = "random"

        # random sampling after Bayesian steps
        # only go in here if user wants to do random sampling after initialization
        elif self.random_step_cadence is not None:

            # determine when random sampling is needed.
            # the first condition picks out iterations at the cadence specified by self.random_step_cadence (using the
            # modulo operator), the second condition makes sure that the first point after random initialization is
            # not picked for randomization
            if (
                (current_iteration - self.num_initial_random_points)
                % self.random_step_cadence
                == 0
            ) & (current_iteration > self.num_initial_random_points):
                sample_type = "random"

        return sample_type

    def random_candidate(self):
        """
        returns a candidate generated via random sampling of the covariates according to the user-specified sampling
        scheme. In first iteration will create required number of random initial candidate datapoints and store in
        attribute "__initial_random_candidates".
        :param
            - self.model["covars_sampled_iter"] (int): number of covariate datapoints sampled so far
            - self.num_initial_random_points (int): number of initial random datapoints (data points obtained from
            random sampling and not from the acquisition function). If set to 0, there's no random start
            - self.random_sampling_method (str): sampling method for random points. Options: "random" and "latin_hcs"
            - self.covar_bounds (tensor, 2 X <num covariates>): upper and lower bounds for covariates provided by user
            - self.device (torch device): computational device used
        :return:
        """

        # get number of covariates
        NUM_COVARS = self.covar_bounds.size()[1]

        # start by creating initial random data point and storing in private attribute candidates at first iteration
        # (if not already stored)
        if (self.model["covars_sampled_iter"] == 0) & (
            not hasattr(self, "__initial_random_candidates")
        ):

            # get candidates by invoking the methods "random" or "latin_hcs" (as carried by
            # self.random_sampling_method)
            # note: are starting from the user-provided initial guess, hence taking
            # n_samp=self.num_initial_random_points-1
            rand_candidates = getattr(DataSamplers, self.random_sampling_method)(
                n_samp=self.num_initial_random_points - 1,
                initial_guess=self.initial_guess,
                covar_bounds=self.covar_bounds,
                device=self.device,
            )
            self.__initial_random_candidates = torch.cat(
                (self.initial_guess, rand_candidates), dim=0
            )

        # self.model["covars_sampled_iter"] counts the already sampled iterations, while the present method is
        # determining sampling type of the NEXT iteration (self.model["covars_sampled_iter"] + 1). Defining local
        # variable to keep track of iteration number for which sampling type is being determined
        current_iteration = self.model["covars_sampled_iter"] + 1

        # CASE 1: create random datapoints from initial random start
        if current_iteration <= self.num_initial_random_points:

            # get the candidate datapoint as a single row in the set of candidates
            # reshape to convert row into a matrix (required through code base)
            candidate = self.__initial_random_candidates[
                current_iteration - 1, :
            ].reshape((1, NUM_COVARS))

        # CASE 2: create random datapoints from interdispersed random points
        else:

            # get candidates by invoking the methods "random" or "latin_hcs" (as carried by
            # self.random_sampling_method)
            candidate = getattr(DataSamplers, self.random_sampling_method)(
                n_samp=1,
                initial_guess=self.initial_guess,
                covar_bounds=self.covar_bounds,
                device=self.device,
            )

        return candidate
