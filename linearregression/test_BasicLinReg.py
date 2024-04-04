import numpy as np
import sys
from BasicLinReg import *

Xlist = []
ylist = []

X = np.array([[1,2],[3,4]])
y = np.array([5,6])

Xlist.append(X)
ylist.append(y)


X = np.array([[1,2,0],[-2,3,4]])
y = np.array([1,-1])

Xlist.append(X)
ylist.append(y)


X = np.array([[1,0],[0,2]])
y = np.array([56,512])

Xlist.append(X)
ylist.append(y)


X = np.array([[-5,-2]])
y = np.array([-14])

Xlist.append(X)
ylist.append(y)




X = np.array([[1,-1],[-1,1]])
y = np.array([5,10])

Xlist.append(X)
ylist.append(y)



X = np.array([[1,1],[1,1],[1,1]])
y = np.array([10,20,30])

Xlist.append(X)
ylist.append(y)



X = np.array([[1,2,3,4,5]]).T
y = np.array([10,100,1000,10000,100000])

Xlist.append(X)
ylist.append(y)
 

answers_linsolve =np.array( [
  [6.0207972893961434, 7.810249675906654, 3.1554436208840472e-30],
[np.nan ,np.nan, np.nan],
[262.0534296665472, 515.0533952902359, 0.0],
[np.nan ,np.nan, np.nan],
[np.nan ,np.nan, np.nan],
[np.nan ,np.nan, np.nan],
[9876.545454545454, 73246.42145773154, 947194368.7272727],
])
                             
answers_gd = np.array([
  [6.013764113048634, 7.809523922688601, 3.4947571000567086e-06],
[0.5705974021574538, 1.4142135623730918, 7.518830502887769e-30],
[262.0534296665464, 515.0533952902344, 1.18301367879288e-24],
[2.5997347344787256, 13.999999999999998, 3.1554436208840472e-30],
[1.767766952966361, 3.535533905932722, 56.25],
[14.14213562373089, 34.6410161513774, 66.66666666666667],
[9876.545454545447, 73246.42145773149, 947194368.7272727],
])


 


def check_answers(testnum, whichcheck):

    X = Xlist[testnum]
    y = ylist[testnum]

    if whichcheck == 'linsolve':
        try:
            theta = solveLinearSystem(X,y)
            ypred = predict(X,theta)
            err = getMeanSquaredError(y,ypred)
            correct1 = np.array([np.linalg.norm(theta),np.linalg.norm(ypred),err])
        except(np.linalg.LinAlgError):
            correct1 = np.array([np.nan,np.nan,np.nan]) 
        check1a = np.logical_and(np.isnan(answers_linsolve[testnum,0]), np.isnan(correct1[0]))
        check1b = np.less(np.sum(np.abs(answers_linsolve[testnum,:]-correct1)),1.e-10)

        check1 = np.logical_or(check1a,check1b)
        
        return check1


    if whichcheck=='graddesc':
        m,n = X.shape
        theta0 = np.zeros(n)
        stepsize = 0.01
        max_steps = 10000
        theta = run_gradient_descent(X,y,theta0, stepsize, max_steps)
        ypred = predict(X,theta)
        err = getMeanSquaredError(y,ypred)
        correct2 = np.array([np.linalg.norm(theta),np.linalg.norm(ypred),err])


        check2 = np.less(np.sum(np.abs(answers_gd[testnum,:]-correct2)),1.e-10)
        
        return check2
    
if __name__ == "__main__":
    
    testnum =int(sys.argv[1])

    whichcheck =sys.argv[2]
    print(testnum, whichcheck)
    check = check_answers(testnum, whichcheck)
    print(check)
    assert(check)
