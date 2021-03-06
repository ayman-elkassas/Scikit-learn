import numpy as np

from sklearn.impute import SimpleImputer

data = [[1,2,np.nan],
        [3,np.nan,1],
        [5,np.nan,0],
        [np.nan,4,6 ],
        [5,0,np.nan],
        [4,5,5]]

print(data)

imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp = imp.fit(data)

modified_data = imp.transform(data)
print(modified_data)
