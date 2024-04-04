import numpy as np


class Module():
    def __init__(self):
        pass

    def forward(self, s_in):
        pass

    def backward(self, dL_ds_out, s_in):
        pass

    def update_var(self, dL_ds_out, s_in, stepsize):
        pass


class Linear(Module):
    def __init__(self, num_outparam, num_inparam, init_type='random'):
        if init_type == 'random':
            self.W = np.random.randn(num_outparam, num_inparam) * .01
            self.b = np.random.randn(num_outparam) * .01
        elif init_type == 'testcase':
            self.W = np.ones((num_outparam, num_inparam))
            self.b = np.ones((num_outparam))

    def forward(self, s_in):
        return np.dot(self.W, s_in) + self.b

    def backward(self, dL_ds_out, s_in):
        return np.dot(self.W.T, dL_ds_out)

    def update_var(self, dL_ds_out, s_in, stepsize):
        self.W = self.W - np.dot(dL_ds_out, s_in.T) * stepsize
        self.b = self.b - dL_ds_out * stepsize
        return None


class Loss():
    pass


class Sigmoid(Module):
    def __init__(self):
        pass

    def forward(self, s_in):
        return 1.0 / (1 + np.exp(-s_in))

    def backward(self, dL_ds_out, s_in):
        sigmoido = self.forward(s_in)
        return sigmoido * (1 - sigmoido) * dL_ds_out


# f(y,hat y) = -y log(hat y) - (1-y) log(1-hat y)
class BCELoss(Loss):
    def __init__(self):
        pass

    def forward(self, y, yhat):
        y = (y + 1) / 2
        yhat = np.clip(yhat, 1e-4, 1 - 1e-4)
        return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    def backward(self, y, yhat):
        y = (y + 1) / 2
        yhat = np.clip(yhat, 1e-4, 1 - 1e-4)
        return ((1.0 / yhat) * y - (1.0 / (1 - yhat)) * (1 - y)) / y.shape[0]

class MNIST_binaryclass():
    def __init__(self, istest=False):
        if istest:
            self.linear = Linear(1, 784, init_type='testcase')
        else:
            self.linear = Linear(1, 784, init_type='random')
        self.sigmoid = Sigmoid()
        self.loss_fn = BCELoss()

        self.states = [None, None, None]
        self.states_grad = [None, None, None]

    def forward(self, X, y):
        self.states[0] = X
        self.states[1] = self.linear.forward(X)
        self.states[2] = self.sigmoid.forward(self.states[1])
        return self.loss_fn.forward(y, self.states[2]), self.states[2]

    def backward(self, X, y):
        self.states_grad[2] = self.loss_fn.backward(y, self.states[2])
        self.states_grad[1] = self.sigmoid.backward(self.states_grad[2], self.states[1])
        self.states_grad[0] = self.linear.backward(self.states_grad[1], self.states[0])

    def update_params(self, stepsize):
        self.linear.update_var(self.states_grad[1], self.states[0], stepsize)

    # return hard prediction
    def inference(self, X, y):
        yhat = self.sigmoid.forward(self.linear.forward(X))
        return self.loss_fn.forward(y, yhat), np.where(yhat >= 0.5, 1, -1)