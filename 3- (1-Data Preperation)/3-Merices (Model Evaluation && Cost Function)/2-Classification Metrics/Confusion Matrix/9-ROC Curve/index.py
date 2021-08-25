import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# for only binary classification

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)

print(fpr, tpr, thresholds)

# result analysis => when threshold in 1.8 then fpr 0 and tpr 1

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

plt.show()
