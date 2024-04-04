import numpy as np


class Module():
    def __init__(self): 
        pass
    def forward(self,s_in):
        pass
    
    def backward(self, dL_ds_out, s_in):
        pass
    def update_var(self,dL_ds_out,s_in, stepsize):
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
        dW = np.dot(dL_ds_out, s_in.T)
        db = np.sum(dL_ds_out, axis=1)

        self.W = self.W - dW * stepsize
        self.b = self.b - db * stepsize

        return
    
class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_in):
        self.s_out = np.maximum(0, s_in)
        return self.s_out
    
    def backward(self, dL_ds_out, s_in):
        dL_ds_in = dL_ds_out * (s_in > 0)
        return dL_ds_in
        
## multiclass classification
class Loss():
    pass
    

class MSELoss(Loss): #copy pasted from HW3
    def __init__(self):
        pass

    def forward(self, y, yhat):
        return np.mean((y - yhat) ** 2) / 2

    def backward(self, y, yhat):
        return (yhat - y) / y.shape[0]
    

    
class MNIST_2layer_binaryclass():
    def __init__(self, nhidden, istest=False):
        #self.layers = [Linear(100,784,0.01), ReLU(),   Linear(100,100,0.01),   ReLU(), Linear(10,100,0.01)]
        if istest:
            self.linear1 = Linear(nhidden,784, init_type = 'testcase')
        else:
            self.linear1 = Linear(nhidden,784, init_type = 'random')
        self.relu = ReLU()

        
        if istest:
            self.linear2 = Linear(1, nhidden, init_type = 'testcase')
        else:
            self.linear2 = Linear(1, nhidden, init_type = 'random')
            
        self.loss_fn = MSELoss()
        self.states = [None, None, None, None]
        self.states_grad = [None, None, None, None]
        
        
    def forward(self,X,y):
        self.states[3] = X
        self.states[2] = self.linear1.forward(self.states[3])
        self.states[1] = self.relu.forward(self.states[2])
        self.states[0] = self.linear2.forward(self.states[1])
        return self.loss_fn.forward(y, self.states[0]), self.states[0]
    
    def backward(self,X,y):
        self.states_grad[0] = self.loss_fn.backward(y, self.states[0])
        self.states_grad[1] = self.linear2.backward(self.states_grad[0], self.states[1])
        self.states_grad[2] = self.relu.backward(self.states_grad[1], self.states[2])
        self.states_grad[3] = self.linear1.backward(self.states_grad[2], self.states[3])
    
    def update_params(self, stepsize):
        self.linear1.update_var(self.states_grad[2], self.states[3], stepsize)
        self.linear2.update_var(self.states_grad[0], self.states[1], stepsize)
                
    
    def inference(self,X,y):
        loss, yhat = self.forward(X,y)
        yhat = np.sign(yhat)
        return loss, yhat
