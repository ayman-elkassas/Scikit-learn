import pandas as pd
import matplotlib.pyplot as plt

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

# Importing the dataset
dataset = pd.read_csv('satf.csv')
dataset.head(10)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor.score(X_train, y_train)
regressor.score(X_test, y_test)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)
median_absolute_error(y_test, y_pred)