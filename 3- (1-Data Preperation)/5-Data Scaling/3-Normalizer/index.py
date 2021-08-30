from sklearn.preprocessing import Normalizer
X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
#  تستخدم l1  عشان يجيب مجموع كل صف ويشوف كل رقم نسبته كام من المجموع ده
#transformer = Normalizer(norm='l1' )
#  تستخدم l2 لجعل جذر مجموع مربعات كل صف هو القيمة العظمي لاخذ النسبه منها
#transformer = Normalizer(norm='l2' )
# تستخدم max   لجعل القيمة العظمي في كل صف هي القيمة العظمي لاخذ النسبه منها
transformer = Normalizer(norm='max')

print(transformer.fit_transform(X))
