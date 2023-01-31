import numpy as np


class LinearRegression:
    """ 
    Linear Regression 
    w: np.ndarray
    b: float
    """

    def __init__(self):

        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fits a linear regression model to the input data X and y by computing and storing the weights."""

        n = X.shape[0]  # num of rows
        # d = X.shape[1] # num of cols
        X = np.hstack((np.ones((n, 1)), X))  ## add col of 1's at the beginning of X
        y = y.reshape(-1, 1)
        if np.linalg.det(X.T @ X) != 0:
            betas = np.linalg.inv(X.T @ X) @ X.T @ y
            self.b = betas[0]
            self.w = betas[1:].reshape(-1)
        else:
            print("No analytical solution due to singular matrix.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Uses weights of the fitted linear regression model to predict y based on the given X. """

        y_hat = X @ self.w + self.b
        return y_hat


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """ Fits the linear regression model to the given data using gradient descent. """
        # raise NotImplementedError()
        n = X.shape[0]  # num of rows
        d = X.shape[1]  # num of cols

        X = np.hstack((np.ones((n, 1)), X))  ## add col of 1's at the beginning of X
        betas = np.zeros(d + 1)

        for i in np.arange(epochs):
            # take gradient step
            grad = -2 / n * X.T @ y + 2 / n * X.T @ X @ betas
            betas = betas - lr * grad

        self.w = betas[1:]
        self.b = betas[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        # raise NotImplementedError()
        y_hat = X @ self.w + self.b
        return y_hat
