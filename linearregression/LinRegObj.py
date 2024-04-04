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
        return np.dot(self.W, s_in) + np.outer(self.b, np.ones(s_in.shape[1]))

    def backward(self, dL_ds_out, s_in):
        return np.dot(self.W.T, dL_ds_out)

    def update_var(self, dL_ds_out, s_in, stepsize):
        batchsize = s_in.shape[1]
        dW = np.dot(dL_ds_out, s_in.T)
        db = np.sum(dL_ds_out, axis=1)

        self.W = self.W - dW * stepsize
        self.b = self.b - db * stepsize

        return

    ## multiclass classification


class Loss():
    pass


class MSELoss(Loss):
    def __init__(self):
        pass

    def forward(self, y, yhat):
        return np.mean((y - yhat) ** 2, axis=0) / 2

    def backward(self, y, yhat):
        return (yhat - y) / y.shape[0]


class MNIST_binaryclass():
    def __init__(self, istest=False):
        # self.layers = [Linear(100,784,0.01), ReLU(),   Linear(100,100,0.01),   ReLU(), Linear(10,100,0.01)]
        if istest:
            self.linear = Linear(1, 784, init_type='testcase')
        else:
            self.linear = Linear(1, 784, init_type='random')

        self.loss_fn = MSELoss()

        self.states = [None, None]  # contains state variables s_k
        self.states_grad = [None, None]  # contains derivatives d loss / d s_k. Note that d Loss/ ds_0 is not needed.

    def forward(self, X, y):
        self.states[0] = X
        self.states[1] = self.linear.forward(self.states[0])
        loss = self.loss_fn.forward(self.states[1], y)
        self.loss = loss

        yhat = self.states[1]
        return np.mean(loss), yhat

    def backward(self, X, y):
        self.states_grad[1] = self.loss_fn.backward(y, self.states[1])
        self.states_grad[0] = self.linear.backward(self.states_grad[1], self.states[0])

    def update_params(self, stepsize):
        self.linear.update_var(self.states_grad[1], self.states[0], stepsize)

    def inference(self, X, y):
        loss, yhat = self.forward(X, y)
        yhat = np.sign(yhat)
        return loss, yhat