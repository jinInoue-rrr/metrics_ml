# linear regression from scratch

import numpy as np
from scipy import linalg
class LinReg:
    def __init__(self):
        self.beta_ = None

    def fit(self, X,y):
        X = np.array(X)
        y = np.array(y)
        xtx = np.dot(X.T, X)
        xty = np.dot(X.T,y)
        self.beta_ = linalg.solve(xtx,xty)
    
    def pred(self,X):
        if X.ndim == 1:
            X = X.reshape(1,-1)
        X = np.array(X)
        return np.dot(X,self.beta_)

