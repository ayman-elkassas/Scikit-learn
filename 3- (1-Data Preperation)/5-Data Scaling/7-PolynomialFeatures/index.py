import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
print(X)

poly = PolynomialFeatures(degree=2 , include_bias = True)
newData=poly.fit_transform(X)
print(newData)