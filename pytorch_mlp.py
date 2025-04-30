"""
Author: Dr Zhibin Liao
Organisation: School of Computer Science and Mathematical Sciences, University of Adelaide
Date: 03-Apr-2025
Description: This Python script demonstrates PyTorch-based Multi-Layer Perceptron (MLP) model creation.

The script is a part of Assignment 2 made for the course COMP SCI 3007/7059/7659 Artificial Intelligence for the year
of 2025. Public distribution of this source code is strictly forbidden.
"""
import numpy as np
from torch import nn
from torch import optim
import torch
from typing import List


class MLPRegression(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 hidden_dim: List[int] = (200, 500, 100),
                 learning_rate: float = 0.001):
        """
        A simple MLP regression model implemented using PyTorch. This class by default uses the Adam optimizer and the
        Mean Square Error loss function to train the regression model. If you are advanced user who understand what they
        mean, please feel free to modify this class. Please submit the pytorch_mlp.py file should you modify the content.
        Args:
            input_dim: the input dimension of the model
            output_dim: the output dimension of the model
            hidden_dim: a list specifying the number of nodes intended for the hidden layers
            learning_rate: the learning rate of the model
        """
        super(MLPRegression, self).__init__()
        layers = []
        for layer_index, dim in enumerate(hidden_dim):
            if layer_index == 0:
                layers.append(nn.Linear(input_dim, dim))
            else:
                layers.append(nn.Linear(hidden_dim[layer_index-1], dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], output_dim, bias=False))

        self.net = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard NN forward pass (you don't need to call this directly).
        Args:
            x: input to the model in torch Tensor type.

        Returns:
            network output of the model in torch Tensor type.
        """
        return self.net(x)

    def fit_step(self, X: np.ndarray, Y: np.ndarray, W: np.ndarray):
        """
        Train (fit) the network by gradient descent for one feed-forward path and one backward path.
        Args:
            X: a batch of input data in the shape of (N, D), N is the batch size, D is the number of features of X
            Y: the corresponding target in the shape of (N, output_dim), output_dim notes the number of model outputs
            W: binary weight matrix in the shape of (N, output_dim), W determines which target values in the target tensor
               should be trained towards during the training.

        Returns:
            None
        """
        self.net.train()  # switch to training mode
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        W = torch.tensor(W, dtype=torch.float32)

        outputs = self.net(X)  # shape (100, 2)
        loss = self.criterion(outputs, Y)
        loss = (loss * W).sum(dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()

        # Update weights
        self.optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on X
        Args:
            X: a batch of input data in the shape of (N, D), N is the batch size, D is the number of features of X

        Returns:
            y_pred: the predicted output of the network in numpy array format
        """
        self.net.eval()  # switch to evaluation mode
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            y_pred = self.net(X)

        return y_pred.numpy()

    def save_model(self, path: str = None):
        torch.save(self.net.state_dict(), path)

    def load_model(self, path: str = None):
        self.net.load_state_dict(torch.load(path, weights_only=True))


if __name__ == '__main__':

    # this is an example usage of the deep learning MLP network picturing a trivial task, i.e.,
    # predicting the maximum and the average for 4 features. In other words, we train an MLP to approximate
    # the maximum and average functions.
    def make_example_dataset(n=1000, dim=4, masking_value=True):
        # we make a data set of 1000 samples, four features each.
        X = np.random.randn(n, dim)
        # we make the label y as in a size of (10000, 2)
        # the first value of each y is the maximum value of all four features
        # the second value of each y is the mean value of all four features
        y1 = np.max(X, axis=1)
        y2 = np.mean(X, axis=1)
        Y = np.stack([y1, y2], axis=1)

        # We randomly mask out a proportion of y values, pretending those missing values were never observed,
        # and we indicate the missing values as -1.
        # Without a label mask, the network will learn to predict the -1s, which is unintended behaviour.
        # With the mask, we can train the network properly. In your assignment, we will need the mask.
        if masking_value:
            W = (np.random.rand(n, 2) > 0.1) * 1
            Y[W == 0] = -1
        else:
            W = np.ones(Y.shape)
        return X, Y, W

    # sample the training data, with the target masks
    X, Y, W = make_example_dataset(n=100000, dim=4, masking_value=True)
    # sample the validation and test data, unmasked
    X_val, Y_val, _ = make_example_dataset(n=100, dim=4, masking_value=False)
    X_test, Y_test, _ = make_example_dataset(n=100, dim=4, masking_value=False)
    network = MLPRegression(input_dim=4, hidden_dim=[100, 200, 100], output_dim=2)

    iterations = 10000
    batch_size = 100
    for i in range(iterations):
        # compose the minibatch data by randomly sample from the training set
        idx = np.random.permutation(len(X))
        x = X[idx[:batch_size], :]
        y = Y[idx[:batch_size], :]
        w = W[idx[:batch_size], :]
        # fit the network for one step
        network.fit_step(x, y, w)

        # predict and observe performance on the validation set every 10 training iterations
        if i % 100 == 0:
            y_pred = network.predict(X_val)
            print(f'Training iteration #{i}:')
            abs_error = np.mean(np.abs(y_pred[:, 0] - Y_val[:, 0]))
            print('MAE for the 1st value of y: {:0.4f}'.format(abs_error))
            abs_error = np.mean(np.abs(y_pred[:, 1] - Y_val[:, 1]))
            print('MAE for the 2nd value of y: {:0.4f}'.format(abs_error))

    # final prediction on the test set
    y_pred = network.predict(X_test)
    print(f'Final test set prediction:')
    abs_error = np.mean(np.abs(y_pred[:, 0] - Y_test[:, 0]))
    print('MAE for the 1st value of y: {:0.4f}'.format(abs_error))
    abs_error = np.mean(np.abs(y_pred[:, 1] - Y_test[:, 1]))
    print('MAE for the 2nd value of y: {:0.4f}'.format(abs_error))

    # save the network model weights after training
    network.save_model(path='temp.ckpt')

    # load the saved weights to an untrained model
    network2 = MLPRegression(input_dim=4, hidden_dim=[100, 200, 100], output_dim=2)
    network2.load_model(path='temp.ckpt')
    # the prediction of the loaded model should be identical to the trained model using the same dataset
    y_pred2 = network2.predict(X_test)
    assert np.all(y_pred2 == y_pred)






