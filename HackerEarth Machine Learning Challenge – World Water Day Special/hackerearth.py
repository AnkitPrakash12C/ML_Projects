# -*- coding: utf-8 -*-
"""HackerEarth.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VdQcLMTOoIpb4U1ouxbjfXCwQ1GpKwIg
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import os
import time

train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')
sample = pd.read_csv('/content/sample_submission.csv')

train.isnull().sum()

test.isnull().sum()

sample.isnull().sum()

train.head(20)

train.fillna({"Temperature": train["Temperature"].mean()}, inplace=True)
test.fillna({"Temperature": test["Temperature"].mean()}, inplace=True)

train.fillna({"Apartment_Type": train["Apartment_Type"].mode()[0]}, inplace=True)
test.fillna({"Apartment_Type": test["Apartment_Type"].mode()[0]}, inplace=True)

train.fillna({"Income_Level": train["Income_Level"].mode()[0]}, inplace=True)
test.fillna({"Income_Level": test["Income_Level"].mode()[0]}, inplace=True)

train["Appliance_Usage"] = train["Appliance_Usage"].fillna(0)
test["Appliance_Usage"] = test["Appliance_Usage"].fillna(0)

train.fillna({"Amenities": train["Amenities"].mode()[0]}, inplace=True)
test.fillna({"Amenities": test["Amenities"].mode()[0]}, inplace=True)

train["Timestamp"] = pd.to_datetime(train["Timestamp"], format='%d/%m/%Y %H', errors='raise')
test["Timestamp"] = pd.to_datetime(test["Timestamp"], format='%d/%m/%Y %H', errors='raise')

train['Timestamp_Unix'] = train['Timestamp'].astype('int64') // 10**9
test['Timestamp_Unix'] = test['Timestamp'].astype('int64') // 10**9

for df in [train, test]:
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

categorical_cols = ["Apartment_Type", "Income_Level", "Amenities", "Appliance_Usage"]
numerical_cols = ["Temperature", "Humidity", "Water_Price", "Period_Consumption_Index", "Residents", "Guests", "Timestamp_Unix", "Year", "Month", "Day", "Hour", "DayOfWeek"]  # Include time-based features

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).astype(str).unique())
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

scaler = StandardScaler()

for col in numerical_cols:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')

train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

train.isnull().sum()

test.isnull().sum()

X = train.drop(['Water_Consumption', 'Timestamp'], axis=1)
y = train['Water_Consumption']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
score = max(0, 100 - rmse)
print(f"LinearRegression: {rmse}")

print(f"Score: {score}")

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
score = max(0, 100 - rmse)
print(f"Decision Tree: {rmse}")

print(f"Score: {score}")

xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
score = max(0, 100 - rmse)
print(f"Xg_Boost: {rmse}")

print(f"Score: {score}")

xgb_model = XGBRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid,
                                   n_iter=50, scoring='neg_mean_squared_error', cv=3, verbose=2, random_state=42)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_score = random_search.best_score_

best_xgb_model = XGBRegressor(random_state=42, **best_params)
best_xgb_model.fit(X_train, y_train)

y_pred = best_xgb_model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # y_test is actual, y_pred is predicted
score = max(0, 100 - rmse)
print(f"Xg_Boost: {rmse}")

print(f"Score: {score}")

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

score = max(0, 100 - rmse)

print(f"Random Forest RMSE: {rmse}")
print(f"Random Forest Score: {score}")

lgbm_model = LGBMRegressor(random_state=42, force_all_finite=False)
lgbm_model = LGBMRegressor(random_state=42, ensure_all_finite=False)
lgbm_model.fit(X_train, y_train)

y_pred = lgbm_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

score = max(0, 100 - rmse)

print(f"LightGBM RMSE: {rmse}")
print(f"LightGBM Score: {score}")

nn_model = MLPRegressor(random_state=42, max_iter=500, early_stopping=True)
nn_model.fit(X_train, y_train)

y_pred = nn_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

score = max(0, 100 - rmse)

print(f"Neural Network RMSE: {rmse}")
print(f"Neural Network Score: {score}")

test_X = test.copy()

test_X['Timestamp'] = pd.to_datetime(test_X['Timestamp'], format='%d/%m/%Y %H', errors='raise')


if 'Timestamp' in test_X.columns:
    test_X = test_X.drop('Timestamp', axis=1)

test_X = imputer.transform(test_X)
test_X = np.nan_to_num(test_X, posinf=1e10, neginf=-1e10)

predictions = best_xgb_model.predict(test_X)

submission_df = pd.DataFrame({'Timestamp': test['Timestamp'], 'Water_Consumption': predictions})

submission_df['Timestamp'] = submission_df['Timestamp'].dt.strftime('%d/%m/%Y %H')


print(f"Shape of submission_df: {submission_df.shape}")

assert submission_df.shape[0] == 6000

timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f'submission_{timestamp}.csv'

submission_df.to_csv(filename, index=False)

print(f"Predictions saved to {filename}")

if os.path.exists(filename):
    file_size = os.path.getsize(filename)
    print(f"File '{filename}' created with size: {file_size} bytes")
else:
    print(f"Error: File '{filename}' was not created.")