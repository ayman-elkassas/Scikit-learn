from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = [0, 2, 1, 3,5,3]
y_true = [0, 1, 2, 3,5,3]
CM1=confusion_matrix(y_true, y_pred)
print(CM1)
# draw heatmap for CM2
sns.heatmap(CM1,center=True)
plt.show()

# TP TN FP FN
# TP + TN = 4
# TP+TN+FP+FN=6
# with normalize (4/6), default
# without (4)

# the most important value in model
print(accuracy_score(y_true, y_pred)) # fraction of all Trues over everything (performance metric)
print(accuracy_score(y_true, y_pred, normalize=False)) # number of all Trues
