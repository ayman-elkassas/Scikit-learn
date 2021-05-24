# Import Libraries
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------

# Calculating Mean Squared Error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
MSE1=mean_squared_error(y_true, y_pred)

y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]

MSE2=mean_squared_error(y_true, y_pred)
MSE3=mean_squared_error(y_true, y_pred, multioutput='uniform_average')

MSE4=mean_squared_error(y_true, y_pred, multioutput='raw_values')

print(MSE1,MSE2,MSE3,MSE4)
