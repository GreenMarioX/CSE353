import numpy as np

def solveLinearSystem(X,y):
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    return theta


def getMeanSquaredError(y,ypred):
    mse = np.mean((y - ypred) ** 2)
    return mse

def predict(X,theta):
    ypred = X @ theta
    return ypred


def get_grad(X,y,theta):
    grad = X.T @ (X @ theta - y) / len(y)
    return grad

def run_gradient_descent(X,y,theta0, stepsize, max_steps):
    theta = theta0
    for k in range(max_steps):
        theta = theta - stepsize*get_grad(X,y,theta)
    return theta

A = np.array([[1,2],[3,4]])
y = np.array([5,6])
print(solveLinearSystem(A, y))