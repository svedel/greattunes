import torch

@staticmethod
def _find_max_response_value(train_X, train_Y):
    """
    determines best (max) response value max_X across recorded values in train_Y, together with the corresponding
    X values
    :param train_X (torch.tensor)
    :param train_Y (torch.tensor)
    :return max_X (float): the X values corresponding to max_Y
    :return max_Y (float): the maximum Y value recorded
    """

    idmax = train_Y.argmax().item()

    max_X = torch.tensor([train_X[idmax].numpy()], dtype=torch.double)
    max_Y = torch.tensor([train_Y[idmax].numpy()], dtype=torch.double)

    print("max_X")
    print(max_X)

    return max_X, max_Y


def _update_max_response_value(self):
    """
    determines the best (max) response across recorded values of train_Y in class instance. Expects that self.train_X,
    self.train_Y exist
    :output
        - self.best_response_value: append latest observation of best Y value
        - self.covars_best_response_value: append with covariates corresponding to best Y value
    """

    try:
        max_X, max_Y = self._find_max_response_value(self.train_X, self.train_Y)
    except:
        raise Exception("creative_project._best_response._update_max_response_value.py: Missing or unable to process "
                        "one of following attributes: self.train_X, self.train_Y")

    # the general case: append to existing data structures
    if self.covars_best_response_value is not None and self.best_response_value is not None:
        self.covars_best_response_value = torch.cat((self.covars_best_response_value, max_X), dim=0)
        self.best_response_value = torch.cat((self.best_response_value, max_Y), dim=0)

    # initializing: set the first elements
    else:
        self.covars_best_response_value = max_X
        self.best_response_value = max_Y
