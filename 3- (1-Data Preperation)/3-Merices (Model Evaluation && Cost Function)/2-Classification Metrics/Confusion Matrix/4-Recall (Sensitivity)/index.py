from sklearn.metrics import recall_score

y_pred=['a','b','c','a','b','c','a','b','c','a']
y_true=['a','a','b','b','a','b','c','c','b','b']

# Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2
# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)

# RecallScore = recall_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

# average => None each class
# but average='micro'

RecallScore=recall_score(y_true,y_pred,average=None)
RecallScore1=recall_score(y_true,y_pred,average='micro')
print(RecallScore)
print(RecallScore1)
