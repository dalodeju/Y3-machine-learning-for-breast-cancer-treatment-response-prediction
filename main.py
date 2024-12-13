import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV


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


# Convert categorical columns (like strings) into numbers
label_encoded_columns = ['ER', 'HER2', 'Gene']  # Columns to encode
for col in label_encoded_columns:
    if col in X_clean.columns:
        encoder = LabelEncoder()
        X_clean[col] = encoder.fit_transform(X_clean[col])  # Replace the original column with numbers
    else:
        print(f"Warning: Column '{col}' not found, skipping encoding.")

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


# Feature Selection

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


# With the following function we can select highly correlated features
# it will remove the first feature that is correlated with any other feature
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

# Classification for PCR
print(X_pca_final.head())

# Ensuring no NaN values in the targets before training
y_pcr_clean = y_pcr_clean.dropna()
X_final = X_final.loc[y_pcr_clean.index]  # Ensure feature set aligns with cleaned target

# Separate the target (pCR) and features again after ensuring no NaNs
features = X_final.drop(columns=['pCR', 'RelapseFreeSurvival','ER','HER2','Gene'])
target = X_final['pCR']

# Separate dataset into train and test again to align with cleaned targets
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=0)


# Training CLASSIFICATION Models

# Decision Tree Model

# Define parameter grid for tuning
print("Performing grid search for parameters tuning in Decision Tree.. \n")
param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced']
}

# Perform grid search
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, scoring='balanced_accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Get the best model
best_tree = grid_search.best_estimator_

dt_pred = best_tree.predict(X_test)
dt_bal_acc = balanced_accuracy_score(y_test, dt_pred)

print("\nBest Decision Tree Parameters:", grid_search.best_estimator_.get_params())
print(f"\nDecision Tree Balanced Accuracy: {dt_bal_acc:.2f}\n")


# Random Forest Model

# Grid search for Random Forest
print("Performing grid search for parameters tuning in Random Forest.. \n")
rf_param_grid = {
    'n_estimators': [50, 100, 200],          # Number of trees
    'max_depth': [None, 10, 20],            # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],        # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]           # Minimum samples required at a leaf node
}

rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    scoring='balanced_accuracy',
    cv=5
)

rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_

rf_pred = best_rf.predict(X_test)
rf_bal_acc = balanced_accuracy_score(y_test, rf_pred)

print("\nBest Random Forest Parameters:", rf_grid_search.best_estimator_.get_params())
print(f"\nRandom Forest Balanced Accuracy: {rf_bal_acc:.2f}\n")

# Support Vector Machine Model

# Grid search for SVM
print("Performing grid search for parameters tuning in SVM.. \n")
svm_param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel type
    'gamma': ['scale', 'auto']  # Kernel coefficient
}

svm_grid_search = GridSearchCV(
    estimator=SVC(random_state=42),
    param_grid=svm_param_grid,
    scoring='balanced_accuracy',
    cv=5
)

svm_grid_search.fit(X_train, y_train)
best_svm = svm_grid_search.best_estimator_

svm_pred = best_svm.predict(X_test)
svm_bal_acc = balanced_accuracy_score(y_test, svm_pred)
print(f"\nSVM Balanced Accuracy: {svm_bal_acc:.2f}\n")

# Training REGRESSION Models

# Regression for RFS
X_rfs = X_pca_final.drop(columns=['pCR', 'RelapseFreeSurvival'])
y_rfs = X_pca_final['RelapseFreeSurvival']

# Train-test split
X_train_rfs, X_test_rfs, y_train_rfs, y_test_rfs = train_test_split(
    X_rfs, y_rfs, test_size=0.2, random_state=0
)

# Linear Regression
print("\nLinear Regression:")
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_rfs, y_train_rfs)
linear_preds = linear_regressor.predict(X_test_rfs)
linear_mae = mean_absolute_error(y_test_rfs, linear_preds)
print(f"Mean Absolute Error (Linear Regression): {linear_mae}")

# Random Forest Regressor
print("\nRandom Forest Regressor:")
rf_regressor = RandomForestRegressor(random_state=42, n_estimators=100)
rf_regressor.fit(X_train_rfs, y_train_rfs)
rf_preds = rf_regressor.predict(X_test_rfs)
rf_mae = mean_absolute_error(y_test_rfs, rf_preds)
print(f"Mean Absolute Error (Random Forest): {rf_mae}")

# Gradient Boosting Regressor
print("\nGradient Boosting Regressor:")
gb_regressor = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
gb_regressor.fit(X_train_rfs, y_train_rfs)
gb_preds = gb_regressor.predict(X_test_rfs)
gb_mae = mean_absolute_error(y_test_rfs, gb_preds)
print(f"Mean Absolute Error (Gradient Boosting): {gb_mae}")

# Hyperparameter tuning for Gradient Boosting
print("\nHyperparameter tuning for Gradient Boosting:")
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, scoring='neg_mean_absolute_error', cv=3)
grid_search.fit(X_train_rfs, y_train_rfs)

# Best parameters and performance
best_gb_regressor = grid_search.best_estimator_
best_gb_preds = best_gb_regressor.predict(X_test_rfs)
best_gb_mae = mean_absolute_error(y_test_rfs, best_gb_preds)
print(f"Best Gradient Boosting Parameters: {grid_search.best_params_}")
print(f"\nMean Absolute Error (Tuned Gradient Boosting): {best_gb_mae}")
            

print(X_pca_final.head())

