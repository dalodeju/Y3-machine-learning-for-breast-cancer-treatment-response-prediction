import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_excel('TrainDataset2024.xls')

# Show the first few rows to inspect the data
print("First few rows of the dataset:")
print(data.head())

# Separate features and outcomes
X = data.drop(columns=['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'])  # Drop outcome columns and ID column
y_pcr = data['pCR (outcome)']  # Target for classification
y_rfs = data['RelapseFreeSurvival (outcome)']  # Target for regression

# Handle missing data
# Impute categorical columns with the most frequent value
categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[categorical_columns] = imputer_cat.fit_transform(X[categorical_columns])

# Impute numerical columns with the mean
numerical_columns = X.select_dtypes(include=[np.number]).columns
if len(numerical_columns) > 0:
    imputer_num = SimpleImputer(strategy='mean')
    X[numerical_columns] = imputer_num.fit_transform(X[numerical_columns])

# Check for missing values after imputation
print("Missing values after imputation:")
print(X.isnull().sum())

# Remove outliers using the Z-score method
z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).all(axis=1)  # Marks rows where all columns exceed Z-score threshold
X_clean = X[~outliers]
y_pcr_clean = y_pcr[~outliers]
y_rfs_clean = y_rfs[~outliers]

print(f"Shape of the cleaned dataset (no outliers): {X_clean.shape}")

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean.select_dtypes(include=[np.number]))

# Perform PCA and retain the first 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Encode categorical features using LabelEncoder
label_encoded_columns = ['ER', 'HER2', 'Gene']
for col in label_encoded_columns:
    if col in X_clean.columns:
        encoder = LabelEncoder()
        X_clean[col] = encoder.fit_transform(X_clean[col])
    else:
        print(f"Warning: Column '{col}' not found for Label Encoding.")

# Combine PCA-transformed data and encoded categorical columns
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
X_encoded = pd.concat([X_pca_df, X_clean[categorical_columns]], axis=1)

# Final dataset for further processing
X_final = pd.concat([X_encoded, X_clean[label_encoded_columns]], axis=1)
X_final['pCR'] = y_pcr_clean
X_final['RelapseFreeSurvival'] = y_rfs_clean

# Show first few rows of the final preprocessed data
print("First few rows of the preprocessed data:")
print(X_final.head())
