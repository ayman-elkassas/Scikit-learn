from sklearn.metrics import f1_score

y_pred = [0, 2, 1, 0, 0, 1]
y_true = [0, 1, 2, 0, 1, 2]

# f1_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)
# average can be : binary,macro, weighted,samples

FS=f1_score(y_true, y_pred, average='micro')

print(FS)