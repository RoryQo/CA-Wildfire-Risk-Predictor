# Predictive Model for Wildfire Property Damage

This project focuses on building a predictive model to assess the risk of wildfire damage to properties in the Los Angeles (L.A.) area. The goal is to classify properties as either at risk for severe wildfire damage (greater than 50%) or not, based on various property characteristics. These features include geographic location, building materials, and fire mitigation elements. The model aims to provide insights into which factors most influence wildfire vulnerability, helping stakeholders such as insurance companies, homeowners, and local governments to better prepare for and mitigate potential wildfire damage.

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

### 2. Classification Selection

Two machine learning models were considered for this task:

- **Naïve Bayes**: A probabilistic model based on Bayes' Theorem, suitable for categorical prediction tasks. It works by calculating the probability of an outcome based on input features.
  
- **k-Nearest Neighbors (kNN)**: A non-parametric, instance-based learning algorithm. It predicts the class of a property based on its proximity to other properties in the feature space.

### 3. Model Selection

Through experimentation, **k-Nearest Neighbors (kNN)** was selected as the optimal model for this problem. The kNN model showed:
- Higher average accuracy compared to Naïve Bayes.
- Lower variance in performance across different cross-validation folds, indicating more stable predictions for new, unseen data.

### 4. Cross-Validation & Hyperparameter Tuning

We employed **grid search** and **cross-validation** to fine-tune the hyperparameters of both models. This ensured that the models were trained on various subsets of the data, and the best performing hyperparameters were selected based on validation results.

Grid search explored various combinations of features and hyperparameters to optimize model performance, ensuring the best possible outcome.

### 5. Error Analysis

The final kNN model demonstrated a balanced error distribution between **Type 1 errors** (false positives) and **Type 2 errors** (false negatives). This balance is crucial for use in real-world applications, particularly in **insurance**. For example, an equal number of false positives and false negatives means that insurance companies can expect no disproportionate risk of over- or under-payouts. This balance helps maintain fairness in the model’s predictions.

## Key Findings

- **Important Features**: The models consistently selected structural elements such as **eaves**, **exterior siding**, and **attached structures** (e.g., decks, fences) as significant predictors of severe wildfire damage. These features, along with **geographic factors** like **latitude** and **longitude**, emerged as critical factors in assessing fire damage risk.
  
- **Geographic Influence**: A spatial analysis of fire damage revealed that certain regions, such as **Santa Rosa** and **San Andreas**, experienced higher rates of damage, whereas areas like **Quincy** and **Stony River George** had fewer homes affected by the fire. This geographical clustering indicates that certain areas are more vulnerable due to proximity to wildfire-prone zones, construction material types, and other local factors.

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
