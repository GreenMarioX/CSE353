import numpy as np
import LogReg
import scipy.io as sio


X = np.array([[1,2,3.],[4.,5/2,6/2],[7/2,8/2,9/2],[.1,.3,.1],[-.1,8,10]])
y = np.array([1,-1,1,1,-1])
z = np.array([-.05,-8,0,1.23,4.56,7.89])
theta = np.array([10,-.4,-.02])
              
import sys

s = sys.argv[1]
if s == 'check_sigmoid':
    zz = LogReg.sigmoid(z)
    zz = np.round(zz*10000)/10000
    assert(np.sum(np.equal(zz,[4.875e-01, 3.000e-04, 5.000e-01, 7.738e-01, 9.896e-01 ,9.996e-01]))==6)
    
    
elif s == 'get_class_accuracy':
    
    assert(LogReg.get_class_accuracy(X, y, theta)==.8) 
    

elif s == 'getLossFunction':
    zz = LogReg.getLossFunction(X, y, theta)
    zz = np.round(zz*10000)/10000
    assert(zz==7.86)

elif s == 'getGradient':
    zz = LogReg.getGradient(X, y, theta)
    zz = np.round(zz*10000)/10000
    assert(np.sum(np.equal(zz,[0.7939, 0.5017, 0.6183]))==3)
    
elif s == 'getStochGradient':
    zz = LogReg.getStochGradient(X, y, theta,[1,2,4])
    zz = np.round(zz*10000)/10000
    assert(np.sum(np.equal(zz,[1.3329 ,0.8657, 1.0404]))==3)
