# import libs
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
# -----------------------------------------------------------

# load iris data
DigitsData=load_digits()

# 1797 image each one is (8*8 px => 64 intensity)

# X Data samples
# note any built-in dataset (**.data) is samples and (**.target) is y
X=DigitsData.data
print('X Data is \n',X[:10,:])
print('X shape is \n',X.shape)
print('X features are \n',DigitsData.feature_names)

print("===============================================================")

y=DigitsData.target
print('y Data is \n',y[:140])
print('y shape is \n',y.shape)
# num and classes name
print('y columns are \n',DigitsData.target_names)

print("================================================================")

plt.gray()

for image in range(10):
    print("Images of number : ",image)
    plt.matshow(DigitsData.images[image])
    print("================================================================")
    plt.show()