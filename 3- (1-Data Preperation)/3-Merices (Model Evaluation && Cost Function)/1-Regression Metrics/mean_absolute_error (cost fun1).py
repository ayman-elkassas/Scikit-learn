# Import Libraries
from sklearn.metrics import mean_absolute_error

# ----------------------------------------------------

# Calculating Mean Absolute Error
# MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') # it can be raw_values
# print('Mean Absolute Error Value is : ', MAEValue)

y_target=[3,-0.5,2,7]
y_predict_hx=[2.5,0.0,2,8]

MAE1=mean_absolute_error(y_target,y_predict_hx)
print(MAE1)

y_true=[[0.5,1],[-1,1],[7,-6]]
y_pred=[[0,2],[-1,2],[8,-5]]

MAE2=mean_absolute_error(y_true,y_pred)  # 0.75
print(MAE2)
MAE3=mean_absolute_error(y_true,y_pred,multioutput='uniform_average')  # 0.75 default
print(MAE3)

MAE4=mean_absolute_error(y_true, y_pred, multioutput='raw_values') # each row
print(MAE4)