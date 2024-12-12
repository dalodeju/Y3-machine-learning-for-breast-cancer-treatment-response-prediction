import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor

# Load the training dataset
data = pd.read_excel('TrainDataset2024.xls')

# Preprocess training data
data.replace(999, np.nan, inplace=True)
X = data.drop(columns=['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'])
y_rfs = data['RelapseFreeSurvival (outcome)']

# Handle missing values
categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[categorical_columns] = imputer_cat.fit_transform(X[categorical_columns])

numerical_columns = X.select_dtypes(include=[np.number]).columns
if len(numerical_columns) > 0:
    imputer_num = SimpleImputer(strategy='mean')
    X[numerical_columns] = imputer_num.fit_transform(X[numerical_columns])

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_columns])

# Encode categorical variables
label_encoded_columns = ['ER', 'HER2', 'Gene']
for col in label_encoded_columns:
    if col in X.columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

# Train Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
gb_regressor.fit(X, y_rfs)

# Load the test dataset
test_data = pd.read_excel('FinalTestDataset2024.xls')

# Preprocess test data
test_data.replace(999, np.nan, inplace=True)
test_ids = test_data['ID']
X_test = test_data.drop(columns=['ID'])

# Handle missing values in test data
if len(categorical_columns) > 0:
    X_test[categorical_columns] = imputer_cat.transform(X_test[categorical_columns])
if len(numerical_columns) > 0:
    X_test[numerical_columns] = imputer_num.transform(X_test[numerical_columns])

# Scale numerical features in test data
X_test_scaled = scaler.transform(X_test[numerical_columns])

# Encode categorical variables in test data
for col in label_encoded_columns:
    if col in X_test.columns:
        X_test[col] = encoder.transform(X_test[col])

# Make predictions on the test data
rfs_predictions = gb_regressor.predict(X_test)

# Save predictions to CSV
output = pd.DataFrame({
    'Patient ID': test_ids,
    'Predicted RFS Outcome': rfs_predictions
})
output.to_csv('RFSPrediction.csv', index=False)

print("Predictions saved to RFSPrediction.csv")
