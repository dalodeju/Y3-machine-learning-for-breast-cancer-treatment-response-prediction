import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.read_excel('TrainDataset2024.xls')

# Show the first few rows to inspect the data
print("First few rows of the dataset:")
print(data.head())

# Separate features and outcomes
X = data.drop(columns=['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'])  # Drop outcome columns and ID column
y_pcr = data['pCR (outcome)']  # Target for classification
y_rfs = data['RelapseFreeSurvival (outcome)']  # Target for regression

# Check for categorical columns (object data types)
categorical_columns = X.select_dtypes(include=['object']).columns
print(f"Categorical columns: {categorical_columns}")

# Check if categorical columns exist
if len(categorical_columns) > 0:
    # Impute missing data for categorical columns with mode
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[categorical_columns] = imputer_cat.fit_transform(X[categorical_columns])
else:
    print("No categorical columns found for imputation.")

# Check for numerical columns (float or int types)
numerical_columns = X.select_dtypes(include=[np.number]).columns
print(f"Numerical columns: {numerical_columns}")

# Impute missing data for numerical columns with mean
if len(numerical_columns) > 0:
    imputer_num = SimpleImputer(strategy='mean')
    X[numerical_columns] = imputer_num.fit_transform(X[numerical_columns])
else:
    print("No numerical columns found for imputation.")

# Check for missing values after imputation
print("Check for missing values after imputation:")
print(X.isnull().sum())

# Outlier Detection - Using Z-score method
z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).all(axis=1)

# Remove rows with outliers
X_clean = X[~outliers]
y_pcr_clean = y_pcr[~outliers]
y_rfs_clean = y_rfs[~outliers]

# Verify the shape of the cleaned dataset
print(f"Shape of the cleaned dataset (without outliers): {X_clean.shape}")

# Normalizing the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Convert categorical columns to numerical using Label Encoding
# The features ER, HER2, and Gene are mandatory, so they are encoded but kept for later use
encoder = LabelEncoder()

# Columns that need to be label encoded
label_encoded_columns = ['ER', 'HER2', 'Gene']
for col in label_encoded_columns:
    if col in X_clean.columns:
        X_clean[col] = encoder.fit_transform(X_clean[col])
    else:
        print(f"Warning: Column '{col}' not found for Label Encoding.")

# Saving the preprocessed data
preprocessed_data = pd.DataFrame(X_scaled, columns=X_clean.columns)
preprocessed_data['pCR'] = y_pcr_clean  # Use this for PCR
preprocessed_data['RelapseFreeSurvival'] = y_rfs_clean  # Use this for RFS

# Show first few rows of the preprocessing
print("First few rows of the preprocessed data:")
print(preprocessed_data.head())

