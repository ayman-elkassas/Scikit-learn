from sklearn.impute import SimpleImputer
import numpy as np

# Dataset

data = [[1,2,0],
        [3,0,1],
        [5,0,0],
        [0,4,6],
        [5,0,0],
        [4,5,5]]

print(data)

# zero values is not accepted for you then make data cleaning to resolve missing values

# 1- Data Cleaning
'''
impute.SimpleImputer(missing_values=nan, strategy='mean’, fill_value=None, verbose=0, copy=True)
'''

# np.nan as None in array
# mean , median , most_frequent , constant as strategy
imputedModule=SimpleImputer(missing_values=0,strategy='mean').fit(data)

data_after_clean=imputedModule.transform(data)

print(data_after_clean)

