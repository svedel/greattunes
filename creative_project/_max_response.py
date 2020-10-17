def find_max_response_value(train_X, train_Y):
    """
    determines best (max) response value max_X across recorded values in train_Y, together with the corresponding
    X values
    :param train_X (torch.tensor)
    :param train_Y (torch.tensor)
    :return max_X (float): the X values corresponding to max_Y
    :return max_Y (float): the maximum Y value recorded
    """

    idmax = train_Y.argmax().item()

    max_X = train_X[idmax]
    max_Y = train_Y[idmax]

    return max_X, max_Y