<h1 align="center">Predictive Model for Wildfire Property Damage</h1>


<div>
  <table align="center">
    <tr>
      <td colspan="2" align="center" style="background-color: white; color: black;"><strong>Table of Contents</strong></td>
    </tr>
    <tr>
      <td style="background-color: white; color: black; padding: 10px;">1. <a href="#project-overview" style="color: black;">Project Overview</a></td>
      <td style="background-color: gray; color: black; padding: 10px;">6. <a href="#future-work" style="color: black;">Future Work</a></td>
    </tr>
    <tr>
      <td style="background-color: gray; color: black; padding: 10px;">2. <a href="#key-features" style="color: black;">Key Features</a></td>
      <td style="background-color: white; color: black; padding: 10px;">7. <a href="#installation" style="color: black;">Installation</a></td>
    </tr>
    <tr>
      <td style="background-color: white; color: black; padding: 10px;">3. <a href="#methodology" style="color: black;">Methodology</a></td>
      <td style="background-color: gray; color: black; padding: 10px;">&nbsp;</td>
    </tr>
    <tr>
      <td colspan="2" style="background-color: gray; color: black; padding: 10px;">
        5. <a href="#key-findings" style="color: black;">Key Findings</a>
      </td>
    </tr>
  </table>
</div>


## Project Overview

Wildfires pose a significant threat to both property and human life. In the wake of recent wildfires in California, it has become clear that predictive models based on property features can help identify homes that are most vulnerable. By leveraging machine learning algorithms to analyze key property attributes, this project seeks to build a reliable tool that can predict the likelihood of severe fire damage.

The project uses a variety of input features:
- **Geographic location**: Latitude and longitude.
- **Structural characteristics**: Including roof construction, exterior siding, and fire mitigation measures like vent screens.
- **Additional features**: Attached structures such as decks or fences.

The model developed in this project uses both **Naïve Bayes** and **k-Nearest Neighbors (kNN)** classifiers, comparing their performance to determine the most accurate approach for predicting fire damage. Based on the results, **kNN** was selected as the best-performing model due to its superior accuracy and consistency.

## Key Features

### Target Variable

- **`damage`**: A binary variable representing whether a property sustained severe damage from a wildfire. The values are:
  - `1`: Severe damage (greater than 50% damage)
  - `0`: No or minimal damage

### Predictor Variables

- **Geographic Features**: Latitude and Longitude are crucial in identifying areas at higher risk for wildfire damage. Proximity to known fire-prone areas or dry forests can significantly influence the likelihood of a property being damaged.
  
- **Structural Features**:
  - **Roof Construction**: The type of roofing material and its fire resistance can impact how quickly a structure catches fire.
  - **Exterior Siding**: The materials used for the outer walls of the property can also play a role in fire resilience.
  - **Attached Structures**: Features such as decks, fences, or garages can contribute to fire spread.
  - **Fire Mitigation Elements**: Elements such as vent screens, which prevent embers from entering the home, are critical in reducing fire risk.

## Methodology

### 1. Data Preparation

The dataset used for this project contained missing values, which required advanced handling techniques:

- **Imputation**:
  - For **numerical values**, missing entries were filled using **FancyImpute**, a deep learning-based algorithm that leverages patterns in the data to impute missing values with high accuracy.
  - For **categorical values**, missing entries were filled with the **mode** (most frequent value) in the column, which is a simpler but effective strategy for categorical data with limited missingness.
  
- **Column Dropping**: Columns with more than 50% missing data were dropped entirely. These features would have required too many assumptions to impute reliably and could have introduced noise into the model.

```
# Step 1: Drop columns with excessive missingness (>50%)
threshold = 0.5  # Set threshold for missing values
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df_cleaned = df.drop(columns=cols_to_drop)

# Step 2: Impute missing values
# Numerical columns - apply SoftImpute
numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Replace missing values with np.nan for compatibility with SoftImpute
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].replace({None: np.nan})

# Apply SoftImpute to impute missing values in numerical columns
soft_impute_data = SoftImpute().fit_transform(df_cleaned[numeric_cols])

# Replace the imputed values back into the DataFrame
df_cleaned[numeric_cols] = soft_impute_data

# Categorical columns - fill missing values with the mode
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

# Step 3: Drop rows with remaining missing values
df_cleaned = df_cleaned.dropna()
```
### 2. Classification Selection

Two machine learning models were considered for this task:

- **Naïve Bayes**: A probabilistic model based on Bayes' Theorem, suitable for categorical prediction tasks. It works by calculating the probability of an outcome based on input features.
  
- **k-Nearest Neighbors (kNN)**: A non-parametric, instance-based learning algorithm. It predicts the class of a property based on its proximity to other properties in the feature space.

### 3. Model Selection

Through experimentation, **k-Nearest Neighbors (kNN)** was selected as the optimal model for this problem. The kNN model showed:
- Higher average accuracy compared to Naïve Bayes.
- Lower variance in performance across different cross-validation folds, indicating more stable predictions for new, unseen data.

<br>

<img src="https://github.com/RoryQo/CA-Wildfire-Risk-Predictor/blob/main/Visualizations/ModelComparison.jpg" alt="Model Comparison" height="400px">


### 4. Cross-Validation & Hyperparameter Tuning

We employed **grid search** and **cross-validation** to fine-tune the hyperparameters of both models. This ensured that the models were trained on various subsets of the data, and the best performing hyperparameters were selected based on validation results.

Grid search explored various combinations of features and hyperparameters to optimize model performance, ensuring the best possible outcome.

```
# Add feature selection step to pipeline
pipe = Pipeline([
    ("standardizer", standardizer),
    ("feature_selection", SelectKBest(score_func=f_classif)),  # Feature selection
    ("knn", knn)
])

# Create search with parameters and set 5-fold cross-validation
classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(X_train, y_train)
```

### 5. Error Analysis

The final kNN model demonstrated a balanced error distribution between **Type 1 errors** (false positives) and **Type 2 errors** (false negatives). This balance is crucial for use in real-world applications, particularly in **insurance**. For example, an equal number of false positives and false negatives means that insurance companies can expect no disproportionate risk of over- or under-payouts. This balance helps maintain fairness in the model’s predictions.

**Classification Report**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.93      | 0.91   | 0.92     | 9270    |
| 1     | 0.93      | 0.94   | 0.93     | 10776   |
| **Accuracy** | | | **0.93** | **20046** |
| **Macro avg** | 0.93 | 0.92 | 0.93 | 20046 |
| **Weighted avg** | 0.93 | 0.93 | 0.93 | 20046 |


## Key Findings

- **Important Features**: The models consistently selected structural elements such as **eaves**, **exterior siding**, and **attached structures** (e.g., decks, fences) as significant predictors of severe wildfire damage. These features, along with **geographic factors** like **latitude** and **longitude**, emerged as critical factors in assessing fire damage risk.
  
- **Geographic Influence**: A spatial analysis of fire damage revealed that certain regions, such as **Santa Rosa** and **San Andreas**, experienced higher rates of damage, whereas areas like **Quincy** and **Stony River George** had fewer homes affected by the fire. This geographical clustering indicates that certain areas are more vulnerable due to proximity to wildfire-prone zones, construction material types, and other local factors.

 <img src="https://github.com/RoryQo/CA-Wildfire-Risk-Predictor/blob/main/Visualizations/Map.png" alt="Map" height="550px">


## Future Work

### 1. **Geospatial Analysis**:
   Future iterations of the model could incorporate additional geographic factors like terrain type, proximity to forested areas, or historical wildfire data. This could enhance the model's ability to predict risk based on specific environmental factors.

### 2. **Model Enhancements**:
   Exploring alternative machine learning algorithms like **Random Forests**, **Support Vector Machines (SVM)**, or **Gradient Boosting** could lead to improvements in prediction accuracy, especially for high-dimensional data.

### 3. **Feature Expansion**:
   Future work could include more granular structural details such as **roof materials**, **window type**, and the **presence of fire-resistant landscaping**. Expanding the dataset with these features may help refine the predictions even further.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/wildfire-damage-prediction.git
   cd wildfire-damage-prediction
