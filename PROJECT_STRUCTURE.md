# Dementia Diagnosis Project Structure

This document provides a detailed overview of all files in the project and their purposes.

## Core Files

### 1. Data Files
- `dementia_dataset.csv`
  - Primary dataset file (28KB, 375 lines)
  - Contains patient data including:
    - Clinical measurements (MMSE, CDR)
    - Demographic information (Age, Gender, Education)
    - MRI measurements (eTIV, nWBV, ASF)
    - Patient classification (Demented/Nondemented)
  - Used as input for the machine learning models

### 2. Source Code
- `dementia_analysis.py`
  - Main Python script (16KB, 435 lines)
  - Contains:
    - DementiaAnalysis class
    - Data preprocessing pipeline
    - Model training and evaluation
    - Visualization functions
    - Streamlit dashboard implementation
  - Entry point for running the application

### 3. Project Documentation
- `README.md`
  - Project overview and setup instructions
  - Usage guidelines
  - Feature descriptions
  - Model details

- `requirements.txt`
  - Lists project dependencies:
    - pandas>=1.3.0
    - numpy>=1.19.5
    - matplotlib>=3.4.3
    - seaborn>=0.11.2
    - scikit-learn>=0.24.2
    - joblib>=1.0.1
    - streamlit>=1.0.0
    - plotly>=5.3.1

## Generated Files

### 1. Model Files
These files are automatically generated when training models:

- `random_forest_model.joblib`
  - Saved Random Forest classifier
  - Contains trained model parameters
  - Used for making predictions

- `neural_network_model.joblib`
  - Saved Neural Network model
  - Contains trained weights and architecture
  - Used for making predictions

- `scaler.joblib`
  - Saved StandardScaler
  - Contains feature scaling parameters
  - Used for preprocessing new data

### 2. Visualization Files
These files are automatically generated during analysis:

- `correlation_matrix.png`
  - Heatmap showing feature correlations
  - Used in the EDA section of dashboard

- `distribution_plots.png`
  - Feature distributions by group
  - 9 subplots for each feature
  - Shows data patterns and separability

- `eda_visualizations.png`
  - Key feature visualizations:
    1. Age Distribution by Group
    2. MMSE Score Distribution
    3. Education Level by Group
    4. Brain Volume Distribution

- `learning_curves_random_forest.png`
  - Learning curves for Random Forest
  - Shows training vs validation performance
  - Used to assess model learning

- `learning_curves_neural_network.png`
  - Learning curves for Neural Network
  - Shows training vs validation performance
  - Used to assess model learning

- `roc_curves.png`
  - ROC curves for both models
  - Shows classification performance
  - Includes AUC scores

- `confusion_matrices.png`
  - Confusion matrices for both models
  - Shows prediction accuracy breakdown
  - Used for model evaluation

- `feature_importance.png`
  - Bar chart of feature importance
  - Based on Random Forest model
  - Shows most influential features

## File Management

### Running the Project
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run dementia_analysis.py
   ```

### Regenerating Visualizations
To regenerate all visualizations:
1. Delete all .png files
2. Run the application again

### Model Retraining
To retrain models:
1. Delete all .joblib files
2. Run the application again

Note: The application will automatically handle file generation and management during runtime. 