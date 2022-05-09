# linear regression from scratch

import numpy as np
from scipy import linalg
class LinReg:
    def __init__(self):
        self.beta_ = None
        
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y)
        xtx = np.dot(X.T, X)#X^TX
        xty = np.dot(X.T,y)#X^Ty
        self.beta_ = linalg.solve(xtx,xty)#FOCを解く
    
    def pred(self,X):
        if X.ndim == 1:
            X = X.reshape(1,-1)
        X = np.array(X)
        return np.dot(X,self.beta_)#X\beta^hat = y_pred

