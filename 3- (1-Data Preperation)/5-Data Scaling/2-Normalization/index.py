from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
newdata = scaler.fit_transform(data)
print(newdata)

# if you want to change range
# scaler = MinMaxScaler(feature_range = (1,5))
