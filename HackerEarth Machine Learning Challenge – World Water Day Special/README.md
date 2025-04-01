Approach Documentation for Water Consumption Prediction

1. Project Overview
Water scarcity is a critical global challenge, and optimizing household water consumption is essential for conservation. Our goal was to build a machine learning model to predict daily water consumption for households based on historical usage data, household characteristics, and environmental factors.

2. Dataset & Preprocessing
We were provided with:
train.csv - (14000 x 12) - Training dataset with historical water consumption data.
test.csv - (6000 x 11) - Test dataset without the target variable.
sample_submission.csv - Example format for submission.

Handling Missing Values
Categorical Features-(`Apartment_Type`, `Income_Level`, `Amenities`, `Appliance_Usage`): Handled using mode imputation.
Numerical Features-(`Temperature`): Handled using median imputation.

Feature Engineering
Timestamp Transformation: Extracted `hour`, `day`, `weekday`, and `month` to capture time-based usage trends.
Encoding Categorical Variables:
One-hot encoding for 'Apartment_Type' and `Income_Level`.
Label encoding for `Amenities` and `Appliance_Usage`.
Scaling: Normalized numerical features using StandardScaler to improve model performance.

3. Model Selection & Training
We experimented with multiple regression models and evaluated them using **Root Mean Squared Error (RMSE):
Baseline Models: Linear Regression, Decision Tree Regressor.
Advanced Models:
Random Forest Regressor -(performed well due to its ability to capture complex patterns)
Gradient Boosting (XGBoost, LightGBM) -(achieved lower RMSE and handled missing data effectively)
Neural Networks -(tested but had higher training complexity)

The best model was LightGBM, achieving the lowest RMSE on the validation set.

4. Final Predictions & Submission
The trained model was used to generate predictions for the test set.
The results were formatted as per `sample_submission.csv`, ensuring:
The correct index (`Timestamp`) was maintained.
The output column was named `Water_Consumption`.
The final predictions were saved as a CSV file and are ready for submission.

5. Tools Used
Programming Language: Python
Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, Matplotlib, Seaborn
Notebook Environment: Jupyter Notebook / Google Colab

6. Files for Submission
Prediction File(`final_submission.csv`) – Contains the predicted water consumption values.
Source Code(`hackerearth.py` or Jupyter Notebook) – The implementation of data preprocessing, feature engineering, model training, and prediction.
This Approach Documentation- (`approach.txt`).

This document provides an overview of our methodology and ensures clarity on the steps taken for the prediction task. 
