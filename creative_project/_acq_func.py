from botorch.acquisition import ExpectedImprovement


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
            raise Exception("kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: no surrogate model set (self.model['model'] is None)")
        if self.train_Y is None:
            raise Exception("kre8_core.creative_project._acq_func.AcqFunction.set_acq_func: no training data provided (self.train_Y is None")

        if self.acq_func["type"] == "EI":
            self.acq_func["object"] = ExpectedImprovement(model=self.model["model"],
                                                          best_f=self.train_Y.max().item())

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
