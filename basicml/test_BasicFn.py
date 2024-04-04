import numpy as np
from BasicFns import *

y = np.array([-1,1,1,-1,-1,1,-1,1])
yhat_soft = np.array([-2.23,3.24,-2.345,5.21,0.012,1.123,-2.423,10.24])
yhat_hard = np.array([-1,-1,1,1,-1,1,1,1])


z = get_binary_prediction(yhat_soft)

import sys

s = int(sys.argv[1])
if s == 1:
    print(np.sum(z)) 
elif s == 2:
    print(np.mean(z)) 
elif s == 3:
    print(get_misclassification_rate(y,yhat_hard)) 
elif s == 4:
    print(get_accuracy_rate(y,yhat_hard))
elif s == 5:
    print(np.round(get_avr_regression_error(yhat_hard,yhat_soft)*1000)/1000) 
elif s == 6:
    print(np.round(get_RMSE(yhat_hard,yhat_soft)*1000)/1000) 








