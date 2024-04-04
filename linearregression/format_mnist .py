import numpy as np
import scipy.io as  sio


data = sio.loadmat('mnist.mat')
Xtrain = data['trainX'].T
Xtest = data['testX'].T
ytrain = data['trainY'][0,:]
ytest = data['testY'][0,:]
idx_train = np.logical_or(ytrain==0,ytrain==1)
idx_test = np.logical_or(ytest==0,ytest==1)

ytrain = ytrain[idx_train].astype(float)*2-1
Xtrain = Xtrain[:,idx_train].astype(float)
ntrain = Xtrain.shape[1]
ytest = ytest[idx_test].astype(float)*2-1
Xtest = Xtest[:,idx_test].astype(float)

