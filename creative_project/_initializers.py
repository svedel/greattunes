import torch
import creative_project._max_response


class Initializers:
    """
    initializer functions used to initialize data structures etc. These functions are to remain private, i.e.
    non-accessible to the user.

    These methods are kept in a separate class to maintain that they be private in the main
    class (that class being defined in __init__.py.
    """

    def __initialize_from_covars(self, covars):
        """
        initialize covars for modeing. Currently only extract starting guess for each covariate, ASSUME each variable
        is continuous.
        TODO:
            - ALSO CONSIDER CHANGING LENGTH SCALES OF MODEL, ACQ_FUNC WHEN WE START TAKING RANGE OF COVARS INTO ACCOUNT
            - Add support for categorical and integer variables
            - Add support for adding name to each covariate
        :param covars (list of tuples): each entry (tuple) must contain (<initial_guess>, <min>, <max>) for each input
        variable. Currently only allows for continuous variables. Aspiration: Data type of input will be preserved and
        code be adapted to accommodate both integer and categorical variables.
        :return initial_guesses (torch.tensor size 1 X num_covars) (first row of design matrix)
        :return bounds (torch.tensor size 2 X num_covars)
        """

        # verify datatype of covars (pending)

        # extract initial guesses
        guesses = [[g[0] for g in covars]]

        # bounds
        lower_bounds = [g[1] for g in covars]
        upper_bounds = [g[2] for g in covars]

        return (
            torch.tensor(guesses, device=self.device, dtype=self.dtype),
            torch.tensor(
                [lower_bounds, upper_bounds], device=self.device, dtype=self.dtype
            ),
        )

    def __initialize_best_response(self):
        """
        initialize best response. If no data present sets to None, otherwise identifies best response up to each data
        point.
        """

        self.covars_best_response_value = None
        self.best_response_value = None
        first = True

        if self.train_Y is not None:
            for it in range(1, self.train_Y.shape[0] + 1):
                # for monkeypatching during unit testing to work, it is required that the full path is added to this
                # import (monkeypatching this function)
                max_X, max_Y = creative_project._max_response.find_max_response_value(
                    self.train_X[:it, :], self.train_Y[:it]
                )

                if first:
                    self.covars_best_response_value = max_X
                    self.best_response_value = max_Y
                    first = False
                else:
                    self.covars_best_response_value = torch.cat(
                        (self.covars_best_response_value, max_X), dim=0
                    )
                    self.best_response_value = torch.cat(
                        (self.best_response_value, max_Y), dim=0
                    )
