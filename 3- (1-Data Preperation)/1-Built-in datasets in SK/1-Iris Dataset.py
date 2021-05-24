# import libs
from sklearn.datasets import load_iris
# -----------------------------------------------------------

# load iris data
IrisData=load_iris()

# X Data samples
# note any built-in dataset (***.data) is samples and .target is y
X=IrisData.data
print('X Data is \n',X[:10,:])
print('X shape is \n',X.shape)
print('X features are \n',IrisData.feature_names)

print("===============================================================")

y=IrisData.target
print('y Data is \n',y[:140])
print('y shape is \n',y.shape)
# num and classes name
print('y columns are \n',IrisData.target_names)

