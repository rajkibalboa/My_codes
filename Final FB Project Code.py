# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:04:30 2019

@author: Ananth
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble   

=============================================================================== 

"""Reading dataset from the file """
#data load
dataset=pd.read_csv(r'D:\downloadss\Dataset.csv')
dataset.columns=dataset.columns.str.replace(',','_')


#data desc
dataset.head(5)
dataset.describe()
dataset.info()
dataset.columns

dataset.isnull().sum()
dataset.count()

===============================================================================

#Imputation
def null_impute(df):
    for cols in list(df.columns.values):
        if df[cols].isnull().sum() == 0:
            df[cols] = df[cols]
        elif df[cols].dtypes == 'float64' or df[cols].dtypes == 'int64':
            df[cols] = df[cols].fillna(df[cols].mean())
        else:
            df[cols] = df[cols].fillna(df[cols].mode()[0])
            
null_impute(dataset)  


===============================================================================
 """ Plots """

#PairPlot
li=x.columns
print(sns.pairplot(data=dataset,y_vars=li,x_vars='output',size = 5))

#Box plot

dataset.iloc[:,0:1].boxplot(return_type='both',
    patch_artist = True,sym='b+')

dataset.iloc[:,1:2].boxplot()

dataset.iloc[:,2:3].boxplot(return_type='both',
    patch_artist = True,sym='b+')

dataset.iloc[:,4:7].boxplot(return_type='both',
    patch_artist = True,sym='b+')

dataset.iloc[:,7:10].boxplot(return_type='both',
    patch_artist = True,sym='b+')

dataset.iloc[:,10:13].boxplot(return_type='both',
    patch_artist = True,sym='b+')

#weekends
dataset.plot.scatter(x='sat_pub',y='comm24')
dataset.plot.scatter(x='sat_pub',y='comm48') 
dataset.plot.scatter(x='sat_pub',y='comm24_1')#max comm
dataset.plot.scatter(x='sun_pub',y='comm24') 
dataset.plot.scatter(x='sun_pub',y='comm48')
dataset.plot.scatter(x='sun_pub',y='comm24_1')#max comm

#weekdays

dataset.plot.scatter(x='mon_pub',y='comm24')
dataset.plot.scatter(x='mon_pub',y='comm48')
dataset.plot.scatter(x='mon_pub',y='comm24_1')#max comm 

dataset.plot.scatter(x='tue_pub',y='comm24')
dataset.plot.scatter(x='tue_pub',y='comm48')
dataset.plot.scatter(x='tue_pub',y='comm24_1')#max comm 

dataset.plot.scatter(x='wed_pub',y='comm24')
dataset.plot.scatter(x='wed_pub',y='comm48')
dataset.plot.scatter(x='wed_pub',y='comm24_1')#max comm 

dataset.plot.scatter(x='wed_pub',y='comm24')
dataset.plot.scatter(x='wed_pub',y='comm48')
dataset.plot.scatter(x='wed_pub',y='comm24_1')#max comm 

dataset.plot.scatter(x='thu_pub',y='comm24')
dataset.plot.scatter(x='thu_pub',y='comm48')
dataset.plot.scatter(x='thu_pub',y='comm24_1')#max comm 

dataset.plot.scatter(x='fri_pub',y='comm24')
dataset.plot.scatter(x='fri_pub',y='comm48')
dataset.plot.scatter(x='fri_pub',y='comm24_1')#max comm 

#correlation 
corr_mat=dataset.corr()
print(sns.heatmap(dataset.corr()))
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

===============================================================================
#Multicolliniarity using vif
def calculate_vif_(x, thresh=10.0):
    variables = list(range(x.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(x.iloc[:, variables].values, ix)
               for ix in range(x.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + x.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True
    print('Remaining variables:')
    print(x.columns[variables])
    return x.iloc[:, variables]

calculate_vif_(dataset, thresh=10.0)

#Droping the columns - 
dataset=dataset.drop("diff_24_48",axis=1)
dataset=dataset.drop("hrs",axis=1)

#Changing category to type category
dataset["Category"] = dataset["Category"].astype('category')
===============================================================================

#independent and target variables assignment
x=dataset.drop('output',axis=1)
y=dataset['output']
print(x)
print(y)

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#Scaling 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


===============================================================================

# building Multiple Linear Regression model to the Training set
regressor = LinearRegression()
reg=regressor.fit(x_train, y_train)


#coefficient and intercept values
print('coefficient values',regressor.coef_)
print('intercept value',regressor.intercept_)

# Predicting the Test set results
y_pred = regressor.predict(x_test)
y_pred=pd.DataFrame(y_pred)

regressor.score(x_test, y_test)


#error values
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

=================================================================================
#OLS
model = sm.ols(formula='output~likes+Returns+Category+commBase+comm24+comm48+comm24_1+baseTime+shares+mon_pub+tue_pub+wed_pub+thu_pub+fri_pub+sat_pub+sun_base+mon_base+tue_base+wed_base+thu_base+fri_base+sat_base',data=dataset)
fitted = model.fit()
print(fitted.summary())

===============================================================================
#DecisionTree
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print( y_pred)

regressor.score(x_test, y_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

===============================================================================
#RandomForest
model = RandomForestRegressor()

regr=RandomForestRegressor(max_depth=5, random_state=0,
n_estimators=100)

regr.fit(x_train, y_train)

regr.score(x_test, y_test, sample_weight=None)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

===============================================================================
#GradientBoost
params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
model.score(x_test, y_test)



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

===========================================================================
#Feature importance
X = dataset.iloc[:,0:27]  #independent columns
y = dataset.iloc[:,-1]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nsmallest(10).plot(kind='barh')
plt.show()