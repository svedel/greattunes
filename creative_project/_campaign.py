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
