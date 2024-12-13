import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

# Load the training dataset
train_data = pd.read_excel('TrainDataset2024.xls')

# Replace 999 with NaN
train_data.replace(999, np.nan, inplace=True)

# Preprocess the training data
X_train = train_data.drop(columns=['pCR (outcome)', 'RelapseFreeSurvival (outcome)', 'ID'])
y_train = train_data['pCR (outcome)']

categorical_columns = X_train.select_dtypes(include=['object']).columns
numerical_columns = X_train.select_dtypes(include=[np.number]).columns

# Handle missing values
if len(categorical_columns) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X_train[categorical_columns] = imputer_cat.fit_transform(X_train[categorical_columns])

if len(numerical_columns) > 0:
    imputer_num = SimpleImputer(strategy='mean')
    X_train[numerical_columns] = imputer_num.fit_transform(X_train[numerical_columns])

# Remove outliers
z_scores = np.abs(stats.zscore(X_train.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).any(axis=1)
X_train = X_train[~outliers]
y_train = y_train[~outliers]

# Drop NaN values from the target variable
y_train = y_train.dropna()
X_train = X_train.loc[y_train.index]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train[numerical_columns])
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_columns, index=X_train.index)

# Encode categorical variables
label_encoded_columns = ['ER', 'HER2', 'Gene']
for col in label_encoded_columns:
    if col in X_train.columns:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])

# Combine scaled numerical features and encoded categorical features
X_train_combined = pd.concat([X_scaled_df, X_train[label_encoded_columns]], axis=1)

# Perform PCA
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_combined)
X_train_pca_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2', 'PC3'], index=X_train.index)

# Train Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42, criterion='gini',class_weight='balanced',max_depth=3,min_samples_leaf=4,min_samples_split=2)
dt_classifier.fit(X_train_pca_df, y_train)

# Load the test dataset
test_data = pd.read_excel('FinalTestDataset2024.xls')

# Replace 999 with NaN
test_data.replace(999, np.nan, inplace=True)

# Extract patient IDs
patient_ids = test_data['ID']

# Preprocess the test data
X_test = test_data.drop(columns=['ID'])

if len(categorical_columns) > 0:
    X_test[categorical_columns] = imputer_cat.transform(X_test[categorical_columns])

if len(numerical_columns) > 0:
    X_test[numerical_columns] = imputer_num.transform(X_test[numerical_columns])

X_test_scaled = scaler.transform(X_test[numerical_columns])
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numerical_columns, index=X_test.index)

for col in label_encoded_columns:
    if col in X_test.columns:
        encoder = LabelEncoder()
        X_test[col] = encoder.fit_transform(X_test[col])

X_test_combined = pd.concat([X_test_scaled_df, X_test[label_encoded_columns]], axis=1)
X_test_pca = pca.transform(X_test_combined)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2', 'PC3'], index=X_test.index)

# Make predictions
predictions = dt_classifier.predict(X_test_pca_df)

# Save predictions to CSV
output = pd.DataFrame({'PatientID': patient_ids, 'PCR Prediction': predictions})
output.to_csv('PCRPrediction.csv', index=False)
