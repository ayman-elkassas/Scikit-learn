from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# todo:ex1

y_pred = ['a','a','b','b','a','b','a','a','a','a']
y_true  = ['a','b','b','a','b','a','a','b','a','b']

# now let's compute (TP, TN, FP, FN)
# CM squared matrix is  (num of class * num of class) then (a,b) then 2*2
CM1=confusion_matrix(y_true, y_pred)
print(CM1)

# draw heatmap for CM1
sns.heatmap(CM1,center=True)
plt.show()

# todo:ex2 => multi classifier (diagonal is Truth for class that should be in highest form)
y_pred = ['a','b','c','a','b','c','a','b','c','a']
y_true =  ['a','a','b','b','a','b','c','c','b','b']

# CM2 is 3*3 (a,b,c) possible classes
CM2=confusion_matrix(y_true, y_pred)
print(CM2)

# draw heatmap for CM2
sns.heatmap(CM2,center=True)
plt.show()

# todo:ex3
y_pred = [5,8,9,9,8,5,5,9,8,5,9,8]
y_true =  [9,9,8,8,5,5,9,5,8,9,8,5]
CM3=confusion_matrix(y_true, y_pred)
print(CM3)

# draw heatmap for CM2
sns.heatmap(CM3,center=True)
plt.show()

'''
note : performance of model is good if : TP, TN (diagonal) is high vs. FP, FN is low
'''


