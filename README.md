# Machine Learning for Breast Cancer Treatment Response Prediction

This project aims to predict the response to breast cancer treatment using machine learning techniques. The goal is to preprocess and prepare the dataset for model training by handling missing values, removing outliers, and scaling features. The ultimate aim is to build models that can classify treatment response and predict relapse-free survival (RFS) based on various medical and genetic features.

## Dependencies

To run this project, you need to install the following dependencies using pip:

```bash
pip install scikit-learn numpy pandas scipy xlrd
```

## Preprocessing Steps

### 1. **Loading the Dataset**

The dataset is loaded using `pandas.read_excel()` from an Excel file (`TrainDataset2024.xls`). The dataset contains both features and target columns:
- **Target Columns**:
  - `pCR (outcome)`: Used for classification tasks.
  - `RelapseFreeSurvival (outcome)`: Used for regression tasks.
- **Features**: All other columns except the target columns and `ID` are used as input features for model training.

### 2. **Handling Missing Data**

#### Categorical Columns:
Missing values in categorical columns (e.g., strings or labels) are filled using the **most frequent value** (mode). This is done using the `SimpleImputer` from `scikit-learn` with the strategy `most_frequent`.

#### Numerical Columns:
Missing values in numerical columns (e.g., integers or floats) are filled using the **mean**. This is done using the `SimpleImputer` with the strategy `mean`.

### 3. **Outlier Detection and Removal**

Outliers in numerical columns are detected using the **Z-score** method. A Z-score indicates how far a value is from the mean in terms of standard deviations. Rows where all numerical features have a Z-score greater than 3 (in absolute value) are considered outliers and are removed from the dataset. This step helps ensure that extreme values do not negatively affect the model.

### 4. **Feature Scaling**

Numerical features often have varying scales, which can lead to poor model performance. Scaling ensures that all numerical features have a mean of 0 and a standard deviation of 1.

- The `StandardScaler` from `scikit-learn` is used for this purpose.
- The `fit_transform()` method is applied to scale only the numerical features of the cleaned data (`X_clean`).

### 5. **Principal Component Analysis (PCA)**

To reduce dimensionality and capture the most important features:
- **PCA (Principal Component Analysis)** is applied using `PCA(n_components=3)` from `scikit-learn`.
- The first three principal components (`PC1`, `PC2`, `PC3`) are retained for further use.

### 6. **Label Encoding**

Categorical variables are converted into numerical format using label encoding:
- `LabelEncoder` from `scikit-learn` is used to assign a unique integer to each category in the columns `ER`, `HER2`, and `Gene`.
- The original categorical columns are replaced with their numeric representations.

### 7. **Combining Features and Targets**

The final dataset includes:
- The scaled PCA-transformed numerical features (`PC1`, `PC2`, `PC3`).
- Encoded categorical features such as `ER`, `HER2`, and `Gene`.
- The target variables:
  - `pCR` for classification.
  - `RelapseFreeSurvival` for regression.

The processed data is saved in a DataFrame for use in model training.

## **Feature Selection**
In this section, we will explore techniques to identify and select the most important features that contribute to the prediction of the target variables (**pCR** and **RelapseFreeSurvival**). Feature selection helps improve the efficiency and accuracy of the model by removing irrelevant or redundant features.

## **Classification (for pCR prediction)**
This section will cover how to apply machine learning classification techniques to predict the **pCR (pathologic complete response)** outcome. We will explore various classification algorithms, such as Decision Trees, Random Forests, and Support Vector Machines (SVM).

## **Regression (for Relapse-Free Survival prediction)**
In this section, we will implement regression algorithms to predict **Relapse-Free Survival (RFS)**, a continuous variable. We will evaluate different regression models like Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.

## **Model Evaluation**
Model evaluation is crucial to assess the performance of the classification and regression models. In this section, we will use appropriate evaluation metrics like Accuracy, Precision, Recall, and F1-Score for classification tasks, and Mean Squared Error (MSE) and R-Squared for regression tasks.

## **Model Tuning**
To improve the performance of the models, we will explore hyperparameter tuning techniques such as Grid Search and Randomized Search to find the optimal settings for the machine learning algorithms.

## **Conclusion**
In the conclusion, we will summarize the findings from the models, reflect on their performance, and provide recommendations for further research or improvements.

