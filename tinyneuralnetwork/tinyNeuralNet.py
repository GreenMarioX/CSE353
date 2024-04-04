import numpy as np


def sigmoid_fn(s):
    return 1 / (1 + np.exp(-s))
    


#############################


class Module():
    pass
    
    
class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_in):
        self.s_out = np.maximum(0, s_in)
        return self.s_out
    
    def backward(self, dL_ds_out, s_in):
        dL_ds_in = dL_ds_out * (s_in > 0)
        return dL_ds_in
    
    
class Sigmoid(Module):
    def __init__(self):
        pass

    def forward(self, s_in):
        return 1.0 / (1 + np.exp(-s_in))

    def backward(self, dL_ds_out, s_in):
        sigmoido = self.forward(s_in)
        return sigmoido * (1 - sigmoido) * dL_ds_out
        
    
class Linear(Module):
    def __init__(self, num_outparam, num_inparam, weight_std=1):
        self.W = np.random.randn(num_outparam,num_inparam)*weight_std
        self.b = np.random.randn(num_outparam)*weight_std
        self.num_params = num_inparam*num_outparam + num_outparam

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
        
         
    
    
    
#####################################


class Model():
    def __init__(self):
        self.states = [] 
        self.states_grad = [] 
        self.layers = []
    def forward(self,X,y):
        s = X
        self.states = [s]
        for l in self.layers:
            s = l.forward(s)
            self.states.append(s)
        loss = self.loss_fn.forward(y,s)
        self.loss = loss
        
        yhat = s
        return np.mean(loss), yhat
        
    
    def backward(self,X,y):
        sgrad = self.loss_fn.backward(y,self.states[-1])
        self.states_grad = [sgrad]
        for i in range(len(self.states)-2,0,-1):
            sgrad = self.layers[i].backward(sgrad,self.states[i])
            self.states_grad.insert(0,sgrad)
            
            
    def update_params(self, stepsize):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if layer.num_params > 0:
                layer.update_var(self.states_grad[i],self.states[i], stepsize)
                
                
#################################################
## binary classification
class Loss():
    pass
    
    
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
    
    
    

class SimpleReluClassNN(Model):
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
    
############################################

## regression

class MSELoss(Loss):
    def __init__(self):
        pass

    def forward(self, y, yhat):
        return np.mean((y - yhat) ** 2) / 2

    def backward(self, y, yhat):
        return (yhat - y) / y.shape[0]
    
    
class SimpleSigmoidRegressNN(Model):
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
        
        
        
#############################################

class CrossEntropyLoss(Loss):
    def __init__(self):
        pass

    def forward(self, y, yhat):

        loss = -np.log(np.exp(yhat[y==1]) / np.sum(np.exp(yhat), axis=1))
        return np.sum(loss)

    def backward(self, y, yhat):

        grad = -y + np.exp(yhat) / np.sum(np.exp(yhat), axis=1)[:, np.newaxis]
        return grad
    
    
    
class SimpleReluMulticlassNN(Model):
    pass