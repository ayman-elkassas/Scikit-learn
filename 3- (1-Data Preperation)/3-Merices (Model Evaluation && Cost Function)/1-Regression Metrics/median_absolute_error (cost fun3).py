# Import Libraries
from sklearn.metrics import median_absolute_error

# ----------------------------------------------------

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
MAE=median_absolute_error(y_true, y_pred)
print(MAE)
