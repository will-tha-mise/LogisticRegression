"""
Custom Implementation of Logistic Regression for learning purposes.
"""
import math

import numpy as np
import pandas as pd


class CustomLogisticRegression:
    """
    Custom Implementation of Logistic Regression
    """
    def __init__(
        self, fit_intercept: bool = True, l_rate: float = 0.01, n_epoch: int = 100
    ):
        self.fit_intercept: bool = fit_intercept
        self.l_rate: float = l_rate
        self.n_epoch: int = n_epoch
        self.bias = 0
        self.coef_: np.ndarray = np.zeros(1)

    @staticmethod
    def sigmoid(t) -> float:
        """
        :param t:
        :return: Result of applying Sigmoid function to t
        """
        return 1 / (1 + math.exp(-t))

    def predict_proba(self, row, coef_) -> float:
        """
        :param row: Set of variables to pass to the model to predict class probability
        :param coef_: The coefficients of the logistic regression model
        :return: The resulting class probability.
        """
        if self.fit_intercept:
            row = np.concatenate([[1], row])
        else:
            row = np.concatenate([[0], row])

        t = np.dot(row, coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train) -> None:
        """
        Method for fitting the Logistic Regression model using Mean Squared Error as the loss function.
        This does not work very well as the curve is not convex and so local minimum may not be found.
        Included for learning reasons.

        :param X_train: Set of training variables
        :param y_train: Set of classes corresponding to the training variables
        """
        if self.fit_intercept:
            self.coef_ = np.zeros(X_train.shape[1] + 1)
        else:
            self.coef_ = np.zeros(X_train.shape[1])

        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train.values):
                y_hat = self.predict_proba(row, self.coef_)
                if self.fit_intercept:
                    row = np.concatenate([[1], row])
                else:
                    row = np.concatenate([[0], row])
                self.coef_ = (
                    self.coef_
                    - self.l_rate
                    * (y_hat - y_train.iloc[i])
                    * y_hat
                    * (1 - y_hat)
                    * row
                )

    def fit_log_loss(self, X_train, y_train) -> None:
        """
        Method for fitting the Logistic Regression model using Log Loss Error as the loss function.

        :param X_train: Set of training variables
        :param y_train: Set of classes corresponding to the training variables
        """
        if self.fit_intercept:
            self.coef_ = np.zeros(X_train.shape[1] + 1)
        else:
            self.coef_ = np.zeros(X_train.shape[1])

        length = len(X_train)

        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train.values):
                y_hat = self.predict_proba(row, self.coef_)
                if self.fit_intercept:
                    row = np.concatenate([[1], row])
                else:
                    row = np.concatenate([[0], row])
                self.coef_ = (
                    self.coef_
                    - (self.l_rate * (y_hat - y_train.iloc[i]) * row) / length
                )

    def predict(self, X_test, cut_off: int = 0.5) -> np.ndarray:
        """
        Method returns the predicted classes of the set of features passed in

        :param X_test: Set of variables for prediction
        :param cut_off: Probability cut off to determine which class should be assigned
        :return: Array of predicted classes
        """
        predictions = np.zeros(X_test.shape[0])
        for i, row in enumerate(X_test.values):
            y_hat = self.predict_proba(row, self.coef_)
            predictions[i] = 0 if y_hat < cut_off else 1
        return predictions  # predictions are binary values - 0 or 1


def standardize(data_array: np.ndarray) -> np.ndarray:
    """
    Helper function to standardise an array using mean and standard deviation
    :param data_array: Array to be standardised
    :return: Standardised Array
    """
    return (data_array - np.mean(data_array)) / np.std(data_array).copy()
