"""
Methods for running campaigns. These are the key user-facing methods
"""
import torch
from greattunes.data_format_mappings import (
    tensor2pretty_covariate,
    tensor2pretty_response,
)


def auto(self, response_samp_func, max_iter=100, rel_tol=None, rel_tol_steps=None):
    """
    This method executes black-box optimization for cases where sampling function response ("response_samp_func")
    is known. In this case no user interaction is needed during each iteration of the Bayesian optimization.

    Example of how to call:

        # covariates of the model (2 parameters)
        # covariates come as a list of tuples (one per covariate: (<initial_guess>, <min>, <max>))
        covars = [(1, 0, 2), (12, 6, 14)]

        # number of iterations
        max_iter = 50

        # initialize the class
        cls = TuneSession(covars=covars)

        # define the response function
        # x is a two-element vector, with one per covariate in 'covars'
        def f(x):
            return -(6 * x[0] - 2) ** 2 * torch.sin(12 * x[1] - 4)

        # run the auto-method
        cls.auto(response_samp_func=f, max_iter=max_iter)


    :param response_samp_func (function): function generating the response
    (input: (1 x num_covars tensor) -> output: (1x1 tensor))
    :param max_iter (int): max number of iterations of the optimization
    :param: rel_tol (float): limit on relative improvement between iterations sufficient to stop iteration
    :param: rel_tol_steps (int): number of steps of no improvement in relative improvement. If this is set, the
    relative tolerance must remain stable for at least rel_tol_steps number of iterations before considered a
    converged
    """

    # set sampling method
    self.sampling["method"] = "functions"
    self.sampling["response_func"] = response_samp_func

    # relative tolerance - convert to float
    if rel_tol is not None:
        rel_tol = float(rel_tol)

    # number of iterations with tolerance limit
    if rel_tol_steps is not None:
        if not rel_tol_steps > 0:
            raise Exception(
                "greattunes._campaign.auto: 'rel_tol_steps' must be greater than 0 but "
                "received " + str(rel_tol_steps)
            )
        else:
            # convert to int
            rel_tol_steps = int(rel_tol_steps)

    # loop
    it = 0
    while it < max_iter:

        # investigate if solution satisfies relative tolerance conditions (if these are added by the user) and stop
        # iteration if the solution is considered converged by these conditions
        if not self._Validators__continue_iterating_rel_tol_conditions(
            rel_tol=rel_tol, rel_tol_steps=rel_tol_steps
        ):
            break

        # update counter
        it += 1

        print("ITERATION " + str(it) + ":\n\tIdentify new covariate datapoint...")

        # initialize acquisition function (if first time data present, otherwise don't do anything)
        self._AcqFunction__initialize_acq_func()

        # get new datapoint according to acquisition function
        # special case of first iteration.
        candidate = self.identify_new_candidate()  # defined in _acq_func.AcqFunc

        # store candidate, update counters
        if self.proposed_X is None:  # use proposed_X as proxy also for train_X

            # the backend data format
            self.proposed_X = candidate  # keeps track of the proposal made
            self.train_X = candidate

            # the pretty data format for users
            self.x_data = tensor2pretty_covariate(
                train_X_sample=candidate, covar_details=self.covar_details
            )
        else:
            # the backend data format
            self.proposed_X = torch.cat((self.proposed_X, candidate), dim=0)
            self.train_X = torch.cat((self.train_X, candidate), dim=0)

            # the pretty data format for users
            self.x_data = self.x_data.append(
                tensor2pretty_covariate(
                    train_X_sample=candidate, covar_details=self.covar_details
                )
            )

        self.model["covars_proposed_iter"] += 1
        self.model["covars_sampled_iter"] += 1

        print("\tGet response for new datapoint...")

        # get response and store
        response = self._get_and_verify_response_input()
        if self.train_Y is None:

            # the backend data format
            self.train_Y = response

            # the pretty data format for users
            self.y_data = tensor2pretty_response(train_Y_sample=response)
        else:
            # the backend data format
            self.train_Y = torch.cat((self.train_Y, response), dim=0)

            # the pretty data format for users
            self.y_data = self.y_data.append(
                tensor2pretty_response(train_Y_sample=response)
            )

        self.model["response_sampled_iter"] += 1

        # update surrogate model
        # self.nu is None except for case where self.model["model_type"] = "SimpleCustomMaternGP", however is not called for any
        # other case
        model_retrain_success_str = self._set_GP_model(nu=self.nu)
        print("\t" + model_retrain_success_str + "...")

        print("\tFinish iteration...")

        # update best response value and associated covariates
        self._update_max_response_value()


def ask(self):
    """
    This 'ask' method together with 'tell' method is how to invoke the framework for cases where the system response is
    not available as a function. Examples of this include when the response is a result of a physical action (e.g. an
    experiment), if a good model does not exist, or if model evaluation is time consuming.

    Example of how to call:

        # covariates of the model (2 parameters)
        # covariates come as a list of tuples (one per covariate: (<initial_guess>, <min>, <max>))
        covars = [(1, 0, 2), (12, 6, 14)]

        # initialize the class
        cls = TuneSession(covars=covars)

        # define the response function
        # x is a two-element vector, with one per covariate in 'covars'
        def f(x):
            return -(6 * x[0] - 2) ** 2 * torch.sin(12 * x[1] - 4)

        # === run one iteration ===
        # propose datapoint to sample
        cls.ask()

        # record actually sampled datapoint and the response
        cls.tell()

    'ask' proposes the next covariate datapoint to sample based on data sampled so far, trained surrogate model
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
        - handle case of first iteration where no historical data exists.
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
    This 'tell' method together with 'ask' method is how to invoke the framework for cases where the system response is
    not available as a function. Examples of this include when the response is a result of a physical action (e.g. an
    experiment), if a good model does not exist, or if model evaluation is time consuming.

    Example of how to call:

        # covariates of the model (2 parameters)
        # covariates come as a list of tuples (one per covariate: (<initial_guess>, <min>, <max>))
        covars = [(1, 0, 2), (12, 6, 14)]

        # initialize the class
        cls = TuneSession(covars=covars)

        # define the response function
        # x is a two-element vector, with one per covariate in 'covars'
        def f(x):
            return -(6 * x[0] - 2) ** 2 * torch.sin(12 * x[1] - 4)

        # === run one iteration ===
        # propose datapoint to sample
        cls.ask()

        # record actually sampled datapoint and the response
        cls.tell()

    specifically this 'tell' method samples the covariates and response corresponding to the output made from the
    "ask"-method, ensures that the provided data is valid, and then updates the surrogate model and stores the provided
    data. It is assumed that 'ask' has been run (assumes a request for new datapoint has been made.).

    :input kwargs:
        - 'covar_obs' (torch tensor of size 1 X num_covars or list): provide observed covars data programmatically. If kwarg
        present, will use this approach over manual input
        - 'response_obs'

    assumes:
        - model, likelihood exists
        - next data point x* proposed
    does:
        - samples x*, y*
        - append x* to collection of covariates, y* to observations made
        - refit model
    """

    # get kwargs (these variables will be None if kwarg not present)
    covars = kwargs.get("covar_obs")
    response = kwargs.get("response_obs")

    # sample covariates for the 'candidate' datapoint proposed by .ask-method
    # using manual input, updates train_X and sampling counter (self.model["covars_sampled_iter"])
    self._get_covars_datapoint(covars)

    # get response for the datapoint added in line above
    # using manual input, updates train_Y and sampling counter (self.model["response_sampled_iter"])
    self._get_response_datapoint(response)

    # retrain the GP model
    # updates the prior and likelihood models behind the scenes
    # self.nu is None except for case where self.model["model_type"] = "SimpleCustomMaternGP", however is not called for any other
    # case
    model_retrain_succes_str = self._set_GP_model(nu=self.nu)

    # print to prompt
    print(model_retrain_succes_str)

    # update best response value and associated covariates
    self._update_max_response_value()
