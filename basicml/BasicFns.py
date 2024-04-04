import numpy as np


def get_binary_prediction(soft_label):
    return np.where(soft_label >= 0, 1.0, -1.0)
    
def get_misclassification_rate(y,yhat):
    return np.mean(y != yhat)
    

def get_accuracy_rate(y,yhat):
    return np.mean(y == yhat)

   
def get_avr_regression_error(y,yhat):
    return np.mean((y - yhat) ** 2)
    
    
def get_RMSE(y,yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))