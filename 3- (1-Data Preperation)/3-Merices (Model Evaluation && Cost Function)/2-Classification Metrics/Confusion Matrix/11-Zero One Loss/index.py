from sklearn.metrics import zero_one_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]

# probability of occure mismatch
print(zero_one_loss(y_true, y_pred)) # .25

# number of time occure mismatch
print(zero_one_loss(y_true, y_pred, normalize=False)) # 1
