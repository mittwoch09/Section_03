import pandas as pd 
import numpy as np

# Regression
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Modelling Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

# Validation
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv('london_bike.csv')

# Feature Engineering

data["timestamp"] = pd.to_datetime(data["timestamp"])

data["month"] = data["timestamp"].apply(lambda x:x.month)
data["day"] = data["timestamp"].apply(lambda x:x.day)
data["hour"] = data["timestamp"].apply(lambda x:x.hour)

data = data.drop("timestamp", axis=1)

# One-hot encoding
'''
dummies_w = pd.get_dummies(data["weather_code"], prefix="weather")
data = pd.concat([data,dummies_w], axis=1)
data = data.drop("weather_code", axis=1)

dummies_s = pd.get_dummies(data["season"], prefix="season")
data = pd.concat([data,dummies_s], axis=1)
data = data.drop("season", axis=1)
'''
# Splitting & scaling the data

X = data.drop("cnt", axis=1)
y = data["cnt"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

# Feature Scaling

cols = X_train.columns

sc = StandardScaler()

X_train = pd.DataFrame(sc.fit_transform(X_train), columns=cols)
X_test = pd.DataFrame(sc.transform(X_test), columns=cols)


# Linear Regression

model_lr = LinearRegression()
model_lr.fit(X_train , y_train)

accuracies = cross_val_score(estimator = model_lr, X = X_train, y = y_train, cv = 5)
y_pred = model_lr.predict(X_test)

print('')
print('####### Linear Regression #######')
print('Score : %.4f' % model_lr.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

# XGBRegressor

model_xgbr = XGBRegressor( n_estimators=1000, objective='reg:squarederror', learning_rate=0.2, n_jobs=-1)
model_xgbr.fit(X_train , y_train)

accuracies = cross_val_score(estimator = model_xgbr, X = X_train, y = y_train, cv = 5)
y_pred = model_xgbr.predict(X_test)

print('')
print('####### XGB Regression #######')
print('Score : %.4f' % model_xgbr.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

# RandomForestRegressor

model_rf = RandomForestRegressor()
model_rf.fit(X_train , y_train)

accuracies = cross_val_score(estimator = model_rf, X = X_train, y = y_train, cv = 5)
y_pred = model_rf.predict(X_test)

print('')
print('####### RandomForest Regression #######')
print('Score : %.4f' % model_rf.score(X_test, y_test))
print(accuracies)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

# Hyperparameter Tuning

n_estimators = [int(x) for x in np.linspace(10,200,10)]
max_depth = [int(x) for x in np.linspace(10,100,10)]
min_samples_split = [2,3,4,5,10]
min_samples_leaf = [1,2,4,10,15,20]
random_grid = {'n_estimators':n_estimators,'max_depth':max_depth,
               'min_samples_split':min_samples_split,'min_samples_leaf':min_samples_leaf}

# RandomForest Regression with RandomizedSearch CV

model_rf_cv = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=model_rf_cv,
                               param_distributions=random_grid,
                               cv = 3)

rf_random.fit(X_train,y_train)
y_pred = rf_random.predict(X_test)

print('')
print('####### RandomForest Regression with RandomizedSearch CV #######')

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)

print('')
print('MSE    : %0.2f ' % mse)
print('MAE    : %0.2f ' % mae)
print('RMSE   : %0.2f ' % rmse)
print('R2     : %0.2f ' % r2)

import pickle

with open('bike_model_rfr.pkl','wb') as pickle_file:
    pickle.dump(model_rf_rscv, pickle_file)
