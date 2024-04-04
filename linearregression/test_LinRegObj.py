import numpy as np
import sys
from LinRegObj import *
import scipy.io as sio

data = sio.loadmat('mini_mnist.mat')
Xtrain = data['Xtrain']
Xtest = data['Xtest']
ytrain = data['ytrain'][0, :]
ytest = data['ytest'][0, :]
ntrain = Xtrain.shape[1]
ntest = Xtest.shape[1]

num_inparam = Xtrain.shape[0]


def check_answers(maxiter):
    model = MNIST_binaryclass(istest=True)

    for iter in range(maxiter):
        model.forward(Xtrain, ytrain)
        model.backward(Xtrain, ytrain)
        model.update_params(stepsize=.0000001)

    loss, yhat = model.inference(Xtrain, ytrain)
    test_loss, test_yhat = model.inference(Xtest, ytest)
    check = np.array([iter, loss / ntrain, np.mean(ytrain == yhat), test_loss / ntest, np.mean(ytest == test_yhat)])
    check = np.round(check * 1000) / 1000
    return check


if __name__ == "__main__":

    maxiter = int(sys.argv[1])

    check = check_answers(maxiter)
    if maxiter == 1:
        #print("Error (test1)\n" + str(np.abs(check - np.array([0.00000000e+00, 2.04083963e+05, 5.38000000e-01, 1.59462045e+05, 5.52000000e-01]))))
        assert (np.sum(np.abs(check - np.array([0.00000000e+00, 2.04083963e+05, 5.38000000e-01, 1.59462045e+05, 5.52000000e-01])) < 1e-5) == 5)
    elif maxiter == 2:
        #print("Error (test2)\n" + str(np.abs(check - np.array([1.0000000e+00, 1.1247761e+05, 5.3800000e-01, 8.9008427e+04, 5.5200000e-01]))))
        assert (np.sum(np.abs(check - np.array([1.0000000e+00, 1.1247761e+05, 5.3800000e-01, 8.9008427e+04, 5.5200000e-01])) < 1e-5) == 5)
    else:
        print("Not a valid check")
        assert (False)