import numpy as np
import sys
from LogRegObj import *
import scipy.io as sio

data = sio.loadmat('mini_mnist.mat')
Xtrain = data['Xtrain']
Xtest = data['Xtest']
ytrain = data['ytrain'][0,:]
ytest = data['ytest'][0,:]
ntrain = Xtrain.shape[1]
ntest = Xtest.shape[1]

num_inparam = Xtrain.shape[0]

def check_answers(maxiter):
    model = MNIST_binaryclass(istest = True)

    for iter in range(maxiter):

        model.forward(Xtrain, ytrain)
        model.backward(Xtrain, ytrain)
        model.update_params(stepsize=.0000001)

    loss, yhat = model.inference(Xtrain, ytrain)
    test_loss, test_yhat = model.inference(Xtest, ytest)
    check = np.array([iter,loss/ntrain, np.mean(ytrain==yhat), test_loss/ntest,np.mean(ytest==test_yhat)])
    check = np.round(check*1000)/1000
    return check


if __name__ == "__main__":

    maxiter =int(sys.argv[1])

    check = check_answers(maxiter)
    print(check)
    if maxiter == 1:

        assert(np.sum(np.abs(check -np.array([0., 0.004, 0.538, 0.004, 0.552])))<.001)
    elif maxiter == 2:
        assert(np.sum(np.abs(check -np.array([1., 0.004, 0.538, 0.004, 0.552])))<.001)
    else:
        print("Not a valid check")
        assert(False)

