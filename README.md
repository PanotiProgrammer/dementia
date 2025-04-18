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

<div align="center">
  <img src="![image](https://github.com/user-attachments/assets/b9c103c4-fa7d-40f9-a0ad-a9694315727d)
" alt="Correlation Matrix of Features" width="600"/>
</div>

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
<div align="center">
  <img src="![Model_performance](https://github.com/user-attachments/assets/fe4c7fbd-b7aa-4625-98aa-d444eb727a06)
" alt="Model Performance Metrics" width="800"/>
</div>

### Feature Importance
<div align="center">
  <img src="![2fafce42b1a4106ef37df2799ca8098bf75c29c903560c9d7036d321](https://github.com/user-attachments/assets/91371855-9cb0-4969-af7c-132424833f62)
" alt="Feature Importance Analysis" width="600"/>
</div>

### Learning Curves
<div align="center">
  <img src="![Uploading learning_curves_random_forest.png…]()
" alt="Random Forest Learning Curve" width="400"/>
  <img src="" alt="Neural Network Learning Curve" width="400"/>
</div>

### Calibration Analysis
<div align="center">
  <img src="![image](https://github.com/user-attachments/assets/cabb4208-7b4c-4d5e-a21e-77aa0d4b47a5)
" alt="Model Calibration Curves" width="600"/>
</div>

### Confusion Matrices
<div align="center">
  <img src="![confusion_matrices](https://github.com/user-attachments/assets/cdad51b9-82e5-4208-9b9a-93aa534f401c)
" alt="Model Confusion Matrices" width="800"/>
</div>

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
  <sub>Built with ❤️ by Anand Sharma</sub>
</div> 
