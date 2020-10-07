import numpy as np
import pytest
import torch

### Fixing state of random number generators for test reproducibility
@pytest.fixture(autouse=True)
def rng_state_tests():
    torch.manual_seed(0)
    np.random.seed(0)


### Simple training data
@pytest.fixture(scope="class")
def custom_models_simple_training_data():
    """
    defines very simple dataset for training of custom GP models. Defined in torch.tensor format
    :return: train_X (torch.tensor)
    :return: train_Y (torch.tensor)
    """
    train_X = torch.tensor([[-1.0]], dtype=torch.double)
    train_Y = torch.tensor([[0.2]], dtype=torch.double)
    return train_X, train_Y
