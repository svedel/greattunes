from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf


class AcqFunction:
    """
    all functionality for acquisition functions
    """

    def set_acq_func(self):
        """
        set the acquisition function
        TODO:
            - assert that model has been initiated (also requires training data)
        """

        if self.model["model"] is None:
            raise Exception(
                "kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: no surrogate model set "
                "(self.model['model'] is None)"
            )
        if self.train_Y is None:
            raise Exception(
                "kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: no training data provided "
                "(self.train_Y is None)"
            )

        LIST_ACQ_FUNCS = ["EI"]
        if not self.acq_func["type"] in LIST_ACQ_FUNCS:
            raise Exception(
                "kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: unsupported acquisition function "
                "name provided. '"
                + self.acq_func["type"]
                + "' not in list of supported acquisition functions ["
                + ", ".join(LIST_ACQ_FUNCS)
                + "]."
            )

        if self.acq_func["type"] == "EI":
            self.acq_func["object"] = ExpectedImprovement(
                model=self.model["model"], best_f=self.train_Y.max().item()
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

        # special case of first iteration.
        # if no training data provided, will start from initial guess provided as part of "covars" in
        # class instance initialization
        if (self.start_from_guess) and (self.model["covars_sampled_iter"] == 0):

            candidate = self.initial_guess

        else:

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
