import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

data = pd.read_excel('TrainDataset2024.xls')

# Show the first few rows
print("First few rows of the dataset:")
print(data.head())

# Replace 999 with NaN
data.replace(999, np.nan, inplace=True)

# Separate the features (X) and the outcomes (y_pcr and y_rfs)
X = data.drop(columns=['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'])  # Drop unwanted columns
y_pcr = data['pCR (outcome)']  # Classification target
y_rfs = data['RelapseFreeSurvival (outcome)']  # Regression target

# Fill missing values in categorical columns (like strings)
categorical_columns = X.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
    # Use the most common value to fill in blanks
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[categorical_columns] = imputer_cat.fit_transform(X[categorical_columns])

# Fill missing values in numerical columns (like numbers)
numerical_columns = X.select_dtypes(include=[np.number]).columns
if len(numerical_columns) > 0:
    # Use the average value to fill in blanks
    imputer_num = SimpleImputer(strategy='mean')
    X[numerical_columns] = imputer_num.fit_transform(X[numerical_columns])

# Check if there are still missing values (should all be 0 now)
print("Missing values after filling:")
print(X.isnull().sum())

# Find rows with extreme values (outliers) and remove them
z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))  # Get Z-scores for numerical data
outliers = (z_scores > 3).all(axis=1)  # True if all values in a row are outliers
X_clean = X[~outliers]  # Keep rows that are NOT outliers
y_pcr_clean = y_pcr[~outliers]
y_rfs_clean = y_rfs[~outliers]

print(f"Shape of the cleaned dataset (no outliers): {X_clean.shape}")

# Scale the numerical data so all columns are on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean.select_dtypes(include=[np.number]))  # Only scale numerical columns
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_columns, index=X_clean.index)

# # Perform PCA and retain the first 3 principal components
# pca = PCA(n_components=3)
# X_pca = pca.fit_transform(X_scaled)

# Convert categorical columns (like strings) into numbers
label_encoded_columns = ['ER', 'HER2', 'Gene']  # Columns to encode
for col in label_encoded_columns:
    if col in X_clean.columns:
        encoder = LabelEncoder()
        X_clean[col] = encoder.fit_transform(X_clean[col])  # Replace the original column with numbers
    else:
        print(f"Warning: Column '{col}' not found, skipping encoding.")

# Combine the PCA-transformed data and the encoded categorical columns
# X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])  # PCA results as a DataFrame
# X_encoded = pd.concat([X_pca_df, X_clean[categorical_columns]], axis=1)  # Add back any other categorical data

# Final preprocessed dataset
X_encoded = pd.concat([X_scaled_df, X_clean[categorical_columns]], axis=1)
X_final = pd.concat([X_encoded, X_clean[label_encoded_columns]], axis=1)  # Add the label-encoded columns
X_final['pCR'] = y_pcr_clean  # Add the pCR outcome column
X_final['RelapseFreeSurvival'] = y_rfs_clean  # Add the RFS outcome column

# Show the first few rows
print("First few rows of the preprocessed data:")
print(X_final.head())

print("\n\n\n Shape of the preprocessed data:")
print(X_final.shape)


#Feature Selection
#Correlation-based technique

# Separate the target (pCR) and features
features = X_final.drop(columns=['pCR', 'RelapseFreeSurvival','ER','HER2','Gene'])
target = X_final['pCR']
saved_columns = X_final[['pCR', 'ER', 'HER2', 'Gene']]  # Save the dropped columns

#Seperate dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=0)

#Ploting, hard to identify, too much features
# import matplotlib.pyplot as plt
# import seaborn as sns
# #Using Pearson Correlation
# plt.figure(figsize=(12,10))
# cor = X_train.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
# plt.show()

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

# Find and drop correlated features
corr_features = correlation(X_train,0.85)
print('Corr features length')
print(len(set(corr_features)))

X_final_selected = X_final.drop(columns=corr_features, axis=1)
print(X_final_selected.head())
print(X_final_selected.shape)

# Perform PCA and retain the first 3 principal components
pca = PCA(n_components=3)

# Apply PCA to the DataFrame excluding specific columns inline
X_for_pca = X_final_selected.drop(columns=['pCR', 'RelapseFreeSurvival', 'ER', 'HER2', 'Gene'])

X_pca = pca.fit_transform(X_for_pca)
X_pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])  # PCA results as a DataFrame
# Merge PCA results back with the excluded columns
X_pca_cat= pd.concat([X_pca_df, X_clean[categorical_columns]], axis=1)  # Add back any other categorical data
X_pca_final = pd.concat([X_pca_cat, X_clean[label_encoded_columns]], axis=1)  # Add the label-encoded columns
X_pca_final['pCR'] = y_pcr_clean  # Add the pCR outcome column
X_pca_final['RelapseFreeSurvival'] = y_rfs_clean  # Add the RFS outcome column

print(X_pca_final.head())
