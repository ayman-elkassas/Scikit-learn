from sklearn.preprocessing import MaxAbsScaler
X = [[ 1., 10., 2.],
     [ 2., 0., 0.],
     [ 5., 1., -1.]]
transformer = MaxAbsScaler()
print(transformer.fit_transform(X))
