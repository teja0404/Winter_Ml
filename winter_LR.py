# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 10:56:01 2019

@author: TEJA
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline





dataset = pd.read_csv(r'winequality.csv')

dataset.shape


dataset.describe()



dataset.isnull().any()






X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH','sulphates','alcohol']].values
y = dataset['quality'].values




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



regressor = LinearRegression()  
regressor.fit(X_train, y_train)




coeff_df = pd.DataFrame(regressor.coef_, dataset.columns[:-1],columns=['Coefficient'])  

coeff_df





y_pred = regressor.predict(X_test)

y_pred=np.around(y_pred)

y_pred

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

df1

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


