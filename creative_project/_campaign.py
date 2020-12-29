"""
Methods for running campaigns. These are the key user-facing methods
"""
import torch


def auto(self, response_samp_func, max_iter=100):
    """
    this method executes black-box optimization for cases where sampling function response ("response_samp_func")
    is known. in this case no user interaction is needed during each iteration of the Bayesian optimization.
    :param response_samp_func (function): function generating the response
    (input: (1 x num_covars tensor) -> output: (1x1 tensor))
    :param max_iter (int): max number of iterations of the optimization
    :TODO:
        - add stopping conditions on error improvement between iterations?
    """

    # set sampling method
    self.sampling["method"] = "functions"
    self.sampling["response_func"] = response_samp_func

    # loop
    it = 0
    while it < max_iter:

        # update counter
        it += 1

        print("ITERATION " + str(it) + ": Identify new covariate datapoint...", end=" ")

        # initialize acquisition function (if first time data present, otherwise don't do anything)
        self._AcqFunction__initialize_acq_func()

        # get new datapoint according to acquisition function
        # special case of first iteration.
        candidate = self.identify_new_candidate()  # defined in _acq_func.AcqFunc

        # store candidate, update counters
        if self.proposed_X is None:  # use proposed_X as proxy also for train_X
            self.proposed_X = candidate
            self.train_X = candidate
        else:
            self.proposed_X = torch.cat((self.proposed_X, candidate), dim=0)
            self.train_X = torch.cat((self.train_X, candidate), dim=0)

        self.model["covars_proposed_iter"] += 1
        self.model["covars_sampled_iter"] += 1

        print("Get response for new datapoint...", end=" ")

        # get response and store
        response = self._get_and_verify_response_input()
        if self.train_Y is None:
            self.train_Y = response
        else:
            # self.train_Y.append(response)
            self.train_Y = torch.cat((self.train_Y, response), dim=0)
        self.model["response_sampled_iter"] += 1

        # update surrogate model
        # self.nu is None except for case where self.model["model_type"] = "Custom", however is not called for any
        # other case
        model_retrain_success_str = self._set_GP_model(nu=self.nu)
        print(model_retrain_success_str + "...", end=" ")

        print("Finish iteration...")

        # update best response value and associated covariates
        self._update_max_response_value()


def ask(self):
    """
    proposes the next covariate datapoint to sample based on data sampled so far, trained surrogate model
    on that data of user-specified type and the chosen acquisition function. Special treatment when starting
    from no knowledge (will start from user-provided guesses in "covars" in class instance initialization).

    Batching is not supported, so proposed datapoint can at most be one iteration ahead of observed data. If
    multiple proposals are attempted, prior un-investigated candidate datapoints (those where proposal is not
    matched with an actual record of observed covariates and response) will be overridden by next request for
    a candidate.

    assumes (general run mode):
        - model, likelihood exists
        - acquisition function exists
    does:
        - retrains acquisition function
        - generate new estimate for x*
        - append x* to record

    special case:
        - handle case of first iteration where no data exists
    """

    # special case where historical data has been added: add a model the first time we iterate through
    if (
        self.train_X is not None
        and self.train_Y is not None
        and self.model["model"] is None
    ):
        # train model
        model_retrain_succes_str = self._set_GP_model(nu=self.nu)
        print(model_retrain_succes_str)

    # initialize acquisition function (if first time data present, otherwise don't do anything)
    self._AcqFunction__initialize_acq_func()

    # generate new candidate covars datapoint
    # special case of first iteration.
    candidate = self.identify_new_candidate()  # defined in _acq_func.AcqFunc

    # remember to print new candidate to prompt in easy-to-read format
    candidate_text_for_display_in_prompt = self._print_candidate_to_prompt(candidate)
    print(candidate_text_for_display_in_prompt)

    # update counters etc
    self._update_proposed_data(candidate)


def tell(self, **kwargs):
    """
    samples the covariates and response corresponding to the output made from the "ask"-method.
    Assumes a request for new datapoint has been made.
    :input kwargs:
        - covars (torch tensor of size 1 X num_covars or list): provide observed covars data programmatically. If kwarg
        present, will use this approach over manual input

    assumes:
        - model, likelihood exists
        - next data point x* proposed
    does:
        - samples x*, y*
        - append x* to collection of covariates, y* to observations made
        - refit model
    """

    # get kwargs (these variables will be None if kwarg not present)
    covars = kwargs.get("covars")

    # sample covariates for the 'candidate' datapoint proposed by .ask-method
    # using manual input, updates train_X and sampling counter (self.model["covars_sampled_iter"])
    self._get_covars_datapoint(covars)

    # get response for the datapoint added in line above
    # using manual input, updates train_Y and sampling counter (self.model["response_sampled_iter"])
    self._get_response_datapoint()

    # retrain the GP model
    # updates the prior and likelihood models behind the scenes
    # self.nu is None except for case where self.model["model_type"] = "Custom", however is not called for any other
    # case
    model_retrain_succes_str = self._set_GP_model(nu=self.nu)

    # print to prompt
    print(model_retrain_succes_str)

    # update best response value and associated covariates
    self._update_max_response_value()
