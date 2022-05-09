import numpy as np
import linear_regression as lr
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
data = load_iris()
covariates = data.data
y = data.target
model = lr.LinReg()
model.fit(X = covariates, y = y)
prediction = model.pred(X = covariates).mean()
print(prediction)




