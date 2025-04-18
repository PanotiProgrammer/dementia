# Dementia Diagnosis Using Machine Learning 🧠

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview
This project implements a robust machine learning system for dementia diagnosis using clinical and MRI data. It combines Random Forest and Neural Network models in an ensemble approach, with advanced calibration and comprehensive validation techniques.

## ✨ Features
- **Advanced Data Preprocessing**
  - Robust scaling of features
  - Sophisticated missing value handling
  - SMOTE for class imbalance correction
  - Feature selection and importance analysis

- **Machine Learning Models**
  - Random Forest Classifier (primary model)
  - Neural Network (secondary model)
  - Ensemble weighting with dynamic adjustment
  - Model calibration using isotonic regression

- **Comprehensive Analysis**
  - Feature importance visualization
  - Calibration curves with confidence intervals
  - ROC curves and confusion matrices
  - Cross-validation with confidence intervals

- **Interactive Dashboard**
  - Real-time predictions
  - Model performance metrics
  - Visualization of results
  - Confidence indicators and warnings

## 📊 Dataset
The project uses the OASIS Brain Dataset, which includes:
- Clinical features (Age, Education, MMSE Score, CDR)
- MRI measurements (nWBV, ASF)
- Demographic information

![Dataset Overview](correlation_matrix.png)

## 🏗️ Model Architecture

### Random Forest
- 200 trees
- Max depth of 10
- Balanced class weights
- Feature subset of 50%
- Isotonic calibration

### Neural Network
- Architectures: [25] or [25,25] neurons
- ReLU/tanh activation
- Adaptive learning rate
- Strong regularization (α = 0.1, 0.5)
- Early stopping

### Ensemble Approach
- Dynamic weighting based on model agreement
- Confidence-based adjustments
- Overconfidence detection
- Disagreement warnings

## 📈 Results

### Model Performance
![Model Performance](model_performance.png)

### Feature Importance
![Feature Importance](feature_importance.png)

### Learning Curves
![Random Forest Learning Curve](learning_curves_random_forest.png)
![Neural Network Learning Curve](learning_curves_neural_network.png)

### Calibration Analysis
![Calibration Curves](calibration_curves.png)

### Confusion Matrices
![Confusion Matrices](confusion_matrices.png)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dementia-diagnosis.git
cd dementia-diagnosis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

Run the Streamlit dashboard:
```bash
streamlit run dementia_analysis.py
```

The dashboard provides:
- Interactive model performance visualization
- Real-time predictions
- Feature importance analysis
- Calibration curves
- Model comparison tools

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Dataset source: OASIS Brain Dataset
- Machine learning frameworks: scikit-learn
- Visualization: Streamlit
- Contributors and maintainers

---

<div align="center">
  <sub>Built with ❤️ by Your Name</sub>
</div> 
