import numpy as np


class LogisticRegressionScratch:
    def __init__(self, n_features, n_classes):
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def one_hot(y, n_classes):
        one_hot_y = np.zeros((len(y), n_classes))
        one_hot_y[np.arange(len(y)), y] = 1
        return one_hot_y

    def forward(self, X):
        logits = X @ self.W + self.b
        return self.softmax(logits)

    def compute_loss(self, X, y):
        n_classes = self.W.shape[1]
        y_onehot = self.one_hot(y, n_classes)
        probs = self.forward(X)
        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))
        return loss

    def compute_gradients(self, X, y):
        m = X.shape[0]
        n_classes = self.W.shape[1]
        y_onehot = self.one_hot(y, n_classes)
        probs = self.forward(X)

        dZ = probs - y_onehot
        dW = (X.T @ dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        return dW, db

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)