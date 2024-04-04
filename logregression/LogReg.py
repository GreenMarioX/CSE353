import numpy as np


def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def get_class_accuracy(X, y, theta):
    predictions = sigmoid(np.dot(X, theta))
    predicted_labels = np.where(predictions >= 0.5, 1, -1)
    accuracy = np.mean(predicted_labels == y)
    return accuracy

def getLossFunction(X, y, theta):
    loss = -np.mean(np.log(sigmoid(y * np.dot(X, theta))))
    return loss


def getGradient(X, y, theta):
    gradient = np.dot(X.T, sigmoid(np.dot(X, theta)) - (y + 1) / 2) / X.shape[0]
    return gradient

def getStochGradient(X, y, theta,idx):
    X = X[idx]
    y = y[idx]
    gradient = np.dot(X.T, sigmoid(np.dot(X, theta)) - (y + 1) / 2) / X.shape[0]
    return gradient



