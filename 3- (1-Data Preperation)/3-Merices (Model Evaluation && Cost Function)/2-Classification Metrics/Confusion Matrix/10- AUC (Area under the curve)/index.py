import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])

# long method
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
print(metrics.auc(fpr, tpr))

# with short way
print(roc_auc_score(y, scores, average='micro'))