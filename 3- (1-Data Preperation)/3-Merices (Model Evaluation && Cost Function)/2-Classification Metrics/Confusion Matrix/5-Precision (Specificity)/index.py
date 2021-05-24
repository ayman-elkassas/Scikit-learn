from sklearn.metrics import precision_score

y_pred=['a','b','c','a','b','c','a','b','c','a']
y_true=['a','a','b','b','a','b','c','c','b','b']

# Calculating Precision Score : (Specificity) #(TP / float(TP + FP))
# precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’,sample_weight=None)
# PrecisionScore = precision_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

# average => None each class
# but average='micro'
score=precision_score(y_true,y_pred,average=None)
score1=precision_score(y_true,y_pred,average='micro')
print(score)
print(score1)