import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                           precision_score, recall_score, f1_score, accuracy_score,
                           precision_recall_curve, average_precision_score, balanced_accuracy_score,
                           brier_score_loss)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import streamlit as st
import os
import sys
from scipy import stats

# Configure matplotlib to use 'Agg' backend
plt.switch_backend('Agg')

# Create output directory for saving files
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_output_path(filename):
    """Get safe output path for saving files"""
    return os.path.join(OUTPUT_DIR, filename)

def calculate_confidence_interval(scores, confidence=0.95):
    """Calculate confidence interval for given scores"""
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)
    ci = stats.t.interval(confidence, n-1, mean, se)
    return mean, ci

def check_data_file(file_path):
    """Check if data file exists and is readable"""
    if not os.path.exists(file_path):
        st.error(f"Error: Data file '{file_path}' not found!")
        return False
    try:
        pd.read_csv(file_path)
        return True
    except Exception as e:
        st.error(f"Error reading data file: {str(e)}")
        return False

@st.cache_data
def load_and_process_data(data_path):
    """Load and process data with caching"""
    try:
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

class DementiaAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = RobustScaler()
        self.best_rf_model = None
        self.best_nn_model = None
        self.cv_results = {}
        self.feature_importance = None
        self.selected_features = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dementia dataset with enhanced feature selection"""
        try:
            self.data = pd.read_csv(self.data_path)
            
            # Handle class imbalance check
            class_counts = self.data['Group'].value_counts()
            st.write("Original class distribution:", class_counts)
            
            # Create a copy of the data
            self.data = self.data.copy()
            
            # Enhanced preprocessing
            # Handle missing values using more sophisticated methods
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if self.data[col].isnull().sum() > 0:
                    # Use median for highly skewed data, mean otherwise
                    if abs(self.data[col].skew()) > 1:
                        self.data[col] = self.data[col].fillna(self.data[col].median())
                    else:
                        self.data[col] = self.data[col].fillna(self.data[col].mean())
            
            # Convert categorical variables
            group_map = {'Demented': 1, 'Nondemented': 0, 'Converted': 1}
            gender_map = {'M': 1, 'F': 0}
            hand_map = {'R': 1, 'L': 0}
            
            self.data['Group'] = self.data['Group'].map(group_map)
            self.data['M/F'] = self.data['M/F'].map(gender_map)
            self.data['Hand'] = self.data['Hand'].map(hand_map)
            
            # Define fixed features - focusing on clinically relevant features
            self.selected_features = ['Age', 'EDUC', 'MMSE', 'CDR', 'nWBV', 'ASF']
            self.X = self.data[self.selected_features].copy()
            self.y = self.data['Group'].copy()
            
            # Remove any remaining rows with NaN values
            mask = ~(self.X.isna().any(axis=1) | self.y.isna())
            self.X = self.X[mask].copy()
            self.y = self.y[mask].copy()
            
            # Split the data with stratification
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.25, random_state=42, stratify=self.y
            )
            
            # Scale the features
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train),
                columns=self.X_train.columns
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test),
                columns=self.X_test.columns
            )
            
            # Apply SMOTE for balanced training data
            smote = SMOTE(random_state=42)
            self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
            self.X_train_balanced = pd.DataFrame(self.X_train_balanced, columns=self.selected_features)
            
            st.write("After SMOTE - Training data distribution:", 
                     pd.Series(self.y_train_balanced).value_counts())
            
        except Exception as e:
            st.error(f"Error in data preprocessing: {str(e)}")
            raise

    def train_models(self):
        """Train and tune Random Forest and Neural Network models with enhanced calibration"""
        try:
            # Define cross-validation strategy
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Random Forest with stable parameters
            rf_param_grid = {
                'n_estimators': [200],
                'max_depth': [10],
                'min_samples_split': [10],
                'min_samples_leaf': [4],
                'max_features': [0.5],
                'class_weight': ['balanced']
            }
            
            rf = RandomForestClassifier(random_state=42)
            rf_grid = GridSearchCV(rf, rf_param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
            rf_grid.fit(self.X_train_balanced, self.y_train_balanced)
            
            # Enhanced calibration for Random Forest
            self.best_rf_model = CalibratedClassifierCV(
                rf_grid.best_estimator_,
                cv=5,
                method='isotonic',
                n_jobs=-1,
                ensemble=True
            )
            self.best_rf_model.fit(self.X_train_balanced, self.y_train_balanced)
            
            # Neural Network with more stable parameters
            nn_param_grid = {
                'hidden_layer_sizes': [(25,), (25,25)],  # Simpler architectures
                'activation': ['relu', 'tanh'],  # More stable activations
                'alpha': [0.1, 0.5],  # Stronger regularization
                'learning_rate': ['adaptive'],
                'early_stopping': [True],
                'validation_fraction': [0.25],
                'shuffle': [True]
            }
            
            nn = MLPClassifier(random_state=42, max_iter=1000)
            nn_grid = GridSearchCV(nn, nn_param_grid, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
            nn_grid.fit(self.X_train_balanced, self.y_train_balanced)
            
            # Enhanced calibration for Neural Network
            self.best_nn_model = CalibratedClassifierCV(
                nn_grid.best_estimator_,
                cv=5,
                method='isotonic',
                n_jobs=-1,
                ensemble=True
            )
            self.best_nn_model.fit(self.X_train_balanced, self.y_train_balanced)
            
            # Calculate feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': rf_grid.best_estimator_.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Perform cross-validation with calibration assessment
            rf_scores = []
            nn_scores = []
            
            for train_idx, val_idx in cv.split(self.X_train_balanced, self.y_train_balanced):
                # Split data
                X_train_cv = self.X_train_balanced.iloc[train_idx]
                y_train_cv = self.y_train_balanced.iloc[train_idx]
                X_val_cv = self.X_train_balanced.iloc[val_idx]
                y_val_cv = self.y_train_balanced.iloc[val_idx]
                
                # Train and calibrate RF
                rf_cv = CalibratedClassifierCV(
                    rf_grid.best_estimator_,
                    cv=3,
                    method='isotonic',
                    ensemble=True
                )
                rf_cv.fit(X_train_cv, y_train_cv)
                rf_scores.append(balanced_accuracy_score(y_val_cv, rf_cv.predict(X_val_cv)))
                
                # Train and calibrate NN
                nn_cv = CalibratedClassifierCV(
                    nn_grid.best_estimator_,
                    cv=3,
                    method='isotonic',
                    ensemble=True
                )
                nn_cv.fit(X_train_cv, y_train_cv)
                nn_scores.append(balanced_accuracy_score(y_val_cv, nn_cv.predict(X_val_cv)))
            
            # Calculate confidence intervals
            rf_mean, rf_ci = calculate_confidence_interval(rf_scores)
            nn_mean, nn_ci = calculate_confidence_interval(nn_scores)
            
            # Store cross-validation results
            self.cv_results = {
                'rf': {'mean': rf_mean, 'ci': rf_ci, 'scores': rf_scores},
                'nn': {'mean': nn_mean, 'ci': nn_ci, 'scores': nn_scores}
            }
            
            # Evaluate models
            self.rf_eval = self.evaluate_model(self.best_rf_model, self.X_test, self.y_test, "Random Forest")
            self.nn_eval = self.evaluate_model(self.best_nn_model, self.X_test, self.y_test, "Neural Network")
            
            # Generate and save plots
            self.plot_calibration_curves()
            self.plot_roc_curves()
            self.plot_confusion_matrices()
            self.visualize_feature_importance()
            
            # Create output directory if it doesn't exist
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
                
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            raise

    def get_weighted_prediction(self, rf_prob, nn_prob):
        """
        Calculate weighted prediction with safeguards against extreme disagreements
        """
        try:
            # Calculate probability difference
            prob_diff = abs(rf_prob - nn_prob)
            
            # If extreme disagreement (>0.5 difference), rely more on RF
            if prob_diff > 0.5:
                weighted_prob = (rf_prob * 0.9) + (nn_prob * 0.1)
            # If moderate disagreement (>0.3 difference), use balanced weights
            elif prob_diff > 0.3:
                weighted_prob = (rf_prob * 0.7) + (nn_prob * 0.3)
            # Otherwise use standard weighting
            else:
                weighted_prob = (rf_prob * 0.85) + (nn_prob * 0.15)
                
            return min(max(weighted_prob, 0), 1)  # Clamp between 0 and 1
            
        except Exception as e:
            st.error(f"Error calculating weighted prediction: {str(e)}")
            return (rf_prob + nn_prob) / 2

    def evaluate_model(self, model, X, y, model_name):
        """Evaluate model with multiple metrics"""
        try:
            # Get predictions
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            balanced_accuracy = accuracy_score(y, y_pred, sample_weight=np.ones(len(y)) / len(y))
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            
            # Calculate sensitivity and specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate positive and negative predictive values
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
        except Exception as e:
            st.error(f"Error in model evaluation: {str(e)}")
            raise

    def plot_calibration_curves(self):
        """Plot enhanced calibration curves for both models with confidence intervals"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Set up the calibration curve parameters
            n_bins = 10  # Fixed number of bins for consistency
            
            # Plot the perfectly calibrated line
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', alpha=0.5)
            
            # Colors for the models
            colors = ['#2ecc71', '#3498db']  # Green for RF, Blue for NN
            
            # Convert test data to numpy arrays and ensure proper shape
            X_test_array = np.asarray(self.X_test)
            y_test_array = np.asarray(self.y_test).ravel()
            
            # Get predictions for both models
            rf_probs = self.best_rf_model.predict_proba(X_test_array)[:, 1]
            nn_probs = self.best_nn_model.predict_proba(X_test_array)[:, 1]
            
            # Calculate and plot for each model
            for probs, name, color in zip(
                [rf_probs, nn_probs],
                ['Random Forest', 'Neural Network'],
                colors
            ):
                # Calculate main calibration curve
                prob_true, prob_pred = calibration_curve(
                    y_test_array, probs,
                    n_bins=n_bins,
                    strategy='uniform'  # Use uniform strategy for consistent bin edges
                )
                
                # Plot the main calibration curve
                plt.plot(prob_pred, prob_true, 's-', color=color, 
                        label=f'{name}', linewidth=2, markersize=8)
                
                # Calculate Brier score
                brier_score = brier_score_loss(y_test_array, probs)
                
                # Bootstrap for confidence intervals
                n_bootstraps = 100
                bootstrap_curves = np.zeros((n_bootstraps, n_bins))
                
                # Perform bootstrapping
                rng = np.random.RandomState(42)
                sample_size = len(y_test_array)
                valid_bootstrap_count = 0
                
                for i in range(n_bootstraps):
                    try:
                        # Bootstrap by sampling with replacement
                        indices = rng.randint(0, sample_size, size=sample_size)
                        
                        if len(np.unique(y_test_array[indices])) < 2:
                            continue
                            
                        # Calculate calibration curve for bootstrap sample
                        bootstrap_prob_true, _ = calibration_curve(
                            y_test_array[indices],
                            probs[indices],
                            n_bins=n_bins,
                            strategy='uniform'  # Use uniform strategy for consistent bin edges
                        )
                        
                        # Store the calibration curve
                        bootstrap_curves[valid_bootstrap_count] = bootstrap_prob_true
                        valid_bootstrap_count += 1
                        
                    except Exception as e:
                        continue
                
                # If we have valid bootstrap samples, calculate and plot confidence intervals
                if valid_bootstrap_count > 0:
                    # Trim to only valid bootstrap samples
                    bootstrap_curves = bootstrap_curves[:valid_bootstrap_count]
                    
                    # Calculate confidence intervals
                    confidence_lower = np.percentile(bootstrap_curves, 2.5, axis=0)
                    confidence_upper = np.percentile(bootstrap_curves, 97.5, axis=0)
                    
                    # Plot confidence intervals
                    plt.fill_between(
                        prob_pred,
                        confidence_lower,
                        confidence_upper,
                        color=color,
                        alpha=0.2,
                        label=f'{name} 95% CI'
                    )
                
                # Add Brier score to plot
                plt.text(0.05, 0.95 - (0.1 * (1 if name == 'Neural Network' else 0)),
                        f'{name} Brier Score: {brier_score:.3f}',
                        transform=plt.gca().transAxes,
                        color=color,
                        fontweight='bold')
            
            # Customize the plot
            plt.title('Calibration Curves with Confidence Intervals', pad=20)
            plt.xlabel('Mean Predicted Probability', fontsize=12)
            plt.ylabel('Observed Fraction of Positives', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left', fontsize=10)
            
            # Add explanatory text
            plt.figtext(0.99, 0.02, 
                       'Curves closer to diagonal indicate better calibration.\n'
                       'Shaded areas show 95% confidence intervals.\n'
                       'Lower Brier scores indicate better calibration.',
                       ha='right', fontsize=8, style='italic')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(get_output_path('calibration_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            st.error(f"Error plotting calibration curves: {str(e)}")
            raise

    def plot_roc_curves(self):
        """Plot ROC curves for both models"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Random Forest ROC
            rf_probs = self.best_rf_model.predict_proba(self.X_test)[:, 1]
            rf_fpr, rf_tpr, _ = roc_curve(self.y_test, rf_probs)
            rf_auc = auc(rf_fpr, rf_tpr)
            plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
            
            # Neural Network ROC
            nn_probs = self.best_nn_model.predict_proba(self.X_test)[:, 1]
            nn_fpr, nn_tpr, _ = roc_curve(self.y_test, nn_probs)
            nn_auc = auc(nn_fpr, nn_tpr)
            plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.grid(True)
            plt.savefig(get_output_path('roc_curves.png'))
            plt.close()
        except Exception as e:
            st.error(f"Error plotting ROC curves: {str(e)}")

    def plot_confusion_matrices(self):
        """Plot confusion matrices for both models"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Random Forest confusion matrix
            rf_pred = self.best_rf_model.predict(self.X_test)
            rf_cm = confusion_matrix(self.y_test, rf_pred)
            sns.heatmap(rf_cm, annot=True, fmt='d', ax=ax1)
            ax1.set_title('Random Forest Confusion Matrix')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # Neural Network confusion matrix
            nn_pred = self.best_nn_model.predict(self.X_test)
            nn_cm = confusion_matrix(self.y_test, nn_pred)
            sns.heatmap(nn_cm, annot=True, fmt='d', ax=ax2)
            ax2.set_title('Neural Network Confusion Matrix')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig(get_output_path('confusion_matrices.png'))
            plt.close()
        except Exception as e:
            st.error(f"Error plotting confusion matrices: {str(e)}")

    def visualize_feature_importance(self):
        """Visualize feature importance from Random Forest model"""
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=self.feature_importance)
            plt.title('Feature Importance from Random Forest Model')
            plt.tight_layout()
            plt.savefig(get_output_path('feature_importance.png'))
            plt.close()
        except Exception as e:
            st.error(f"Error visualizing feature importance: {str(e)}")

    def create_dashboard(self):
        """Create an interactive Streamlit dashboard for model analysis and predictions."""
        try:
            st.title("Dementia Diagnosis Analysis Dashboard")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "Feature Importance", "Calibration", "Make Prediction"])
            
            with tab1:
                st.header("Model Performance Metrics")
                
                # Display cross-validation results
                st.subheader("Cross-Validation Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Random Forest CV Metrics:")
                    if isinstance(self.cv_results, dict) and 'rf' in self.cv_results:
                        st.write(f"Mean Score: {float(self.cv_results['rf']['mean']):.3f}")
                        if 'scores' in self.cv_results['rf']:
                            st.write(f"Standard Deviation: {float(np.std(self.cv_results['rf']['scores'])):.3f}")
                
                with col2:
                    st.write("Neural Network CV Metrics:")
                    if isinstance(self.cv_results, dict) and 'nn' in self.cv_results:
                        st.write(f"Mean Score: {float(self.cv_results['nn']['mean']):.3f}")
                        if 'scores' in self.cv_results['nn']:
                            st.write(f"Standard Deviation: {float(np.std(self.cv_results['nn']['scores'])):.3f}")
                
                # Display test set performance
                st.subheader("Test Set Performance")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.write("Random Forest Test Metrics:")
                    if hasattr(self, 'rf_eval') and isinstance(self.rf_eval, dict):
                        for metric, value in self.rf_eval.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                st.write(f"{metric}: {float(value):.3f}")
                
                with col4:
                    st.write("Neural Network Test Metrics:")
                    if hasattr(self, 'nn_eval') and isinstance(self.nn_eval, dict):
                        for metric, value in self.nn_eval.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                st.write(f"{metric}: {float(value):.3f}")
                
                # Display ROC curves and confusion matrices
                if os.path.exists(get_output_path("roc_curves.png")):
                    st.image(get_output_path("roc_curves.png"), caption="ROC Curves for Both Models")
                
                if os.path.exists(get_output_path("confusion_matrices.png")):
                    st.image(get_output_path("confusion_matrices.png"), caption="Confusion Matrices")
            
            with tab2:
                st.header("Feature Importance Analysis")
                if os.path.exists(get_output_path("feature_importance.png")):
                    st.image(get_output_path("feature_importance.png"), caption="Feature Importance from Random Forest Model")
                    
                    if hasattr(self, 'feature_importance') and isinstance(self.feature_importance, pd.DataFrame):
                        st.dataframe(self.feature_importance.round(3))
            
            with tab3:
                st.header("Model Calibration")
                st.write("""
                Calibration curves show how well the predicted probabilities of models match the actual probabilities.
                A perfectly calibrated model will follow the diagonal line.
                """)
                if os.path.exists(get_output_path("calibration_curves.png")):
                    st.image(get_output_path("calibration_curves.png"), caption="Calibration Curves for Both Models")
            
            with tab4:
                st.header("Make a Prediction")
                st.write("Enter patient information to get a dementia prediction")
                
                # Create input fields for features
                age = st.number_input("Age", min_value=0, max_value=120, value=70)
                educ = st.number_input("Years of Education", min_value=0, max_value=30, value=12)
                mmse = st.number_input("MMSE Score", min_value=0, max_value=30, value=25)
                cdr = st.number_input("Clinical Dementia Rating (CDR)", min_value=0.0, max_value=3.0, value=0.0, step=0.5)
                nwbv = st.number_input("Normalized Whole Brain Volume", min_value=0.0, max_value=1.0, value=0.7)
                asf = st.number_input("Atlas Scaling Factor (ASF)", min_value=0.0, max_value=2.0, value=1.0)
                
                if st.button("Get Prediction"):
                    try:
                        # Prepare input data with exact same features used in training
                        input_data = pd.DataFrame({
                            'Age': [age],
                            'EDUC': [educ],
                            'MMSE': [mmse],
                            'CDR': [cdr],
                            'nWBV': [nwbv],
                            'ASF': [asf]
                        })
                        
                        # Scale the features
                        scaled_input = self.scaler.transform(input_data)
                        
                        # Get predictions from both models
                        rf_prob = float(self.best_rf_model.predict_proba(scaled_input)[0][1])
                        nn_prob = float(self.best_nn_model.predict_proba(scaled_input)[0][1])
                        
                        # Validate predictions
                        if nn_prob == 1.0 or nn_prob == 0.0:
                            st.warning("Neural Network prediction appears overconfident. Results should be interpreted with caution.")
                        
                        if abs(rf_prob - nn_prob) > 0.5:
                            st.warning("Large disagreement between models. Results should be interpreted with caution.")
                        
                        # Get weighted prediction
                        weighted_prob = float(self.get_weighted_prediction(rf_prob, nn_prob))
                        
                        # Display results
                        st.subheader("Prediction Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Random Forest Probability", f"{rf_prob:.3f}")
                        with col2:
                            st.metric("Neural Network Probability", f"{nn_prob:.3f}")
                        with col3:
                            st.metric("Weighted Ensemble Probability", f"{weighted_prob:.3f}")
                        
                        # Add interpretation
                        st.write("---")
                        if weighted_prob >= 0.5:
                            risk_level = "High" if weighted_prob > 0.75 else "Moderate"
                            st.warning(f"{risk_level} risk of dementia detected (Probability: {weighted_prob:.1%})")
                        else:
                            risk_level = "Low" if weighted_prob < 0.25 else "Borderline"
                            st.success(f"{risk_level} risk of dementia detected (Probability: {weighted_prob:.1%})")
                        
                        # Add model agreement information
                        prob_diff = abs(rf_prob - nn_prob)
                        if prob_diff > 0.15:
                            st.info(
                                f"Note: Models show {'significant' if prob_diff > 0.3 else 'moderate'} disagreement "
                                f"(difference: {prob_diff:.1%}). Consider consulting with a healthcare professional."
                            )
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.error("Please ensure all input values are within expected ranges.")
            
        except Exception as e:
            st.error(f"Error creating dashboard: {str(e)}")
            raise e

    def save_models(self):
        """Save the trained models and metadata"""
        try:
            joblib.dump(self.best_rf_model, get_output_path('random_forest_model.joblib'))
            joblib.dump(self.best_nn_model, get_output_path('neural_network_model.joblib'))
            joblib.dump(self.scaler, get_output_path('scaler.joblib'))
            joblib.dump(self.selected_features, get_output_path('selected_features.joblib'))
            self.feature_importance.to_pickle(get_output_path('feature_importance.pkl'))
        except Exception as e:
            st.error(f"Error saving models: {str(e)}")
            raise

    def load_saved_models(self):
        """Load saved models and metadata"""
        try:
            self.best_rf_model = joblib.load(get_output_path('random_forest_model.joblib'))
            self.best_nn_model = joblib.load(get_output_path('neural_network_model.joblib'))
            self.scaler = joblib.load(get_output_path('scaler.joblib'))
            self.selected_features = joblib.load(get_output_path('selected_features.joblib'))
            self.feature_importance = pd.read_pickle(get_output_path('feature_importance.pkl'))
            return True
        except Exception as e:
            st.error(f"Error loading saved models: {str(e)}")
            return False

if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Dementia Diagnosis Analysis",
            page_icon="🧠",
            layout="wide"
        )

        # Check if data file exists
        data_file = 'dementia_dataset.csv'
        if not check_data_file(data_file):
            st.stop()

        # Show loading message
        with st.spinner('Loading and processing data...'):
            try:
                # Initialize analysis
                analysis = DementiaAnalysis(data_file)
                
                # Load and preprocess data first
                analysis.load_and_preprocess_data()
                
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Always train new models for consistency
                st.info('Training models with optimized parameters...')
                
                try:
                    # Train models
                    progress_bar.progress(40)
                    st.text('Training machine learning models...')
                    analysis.train_models()
                    
                    # Save models
                    progress_bar.progress(80)
                    st.text('Saving models...')
                    analysis.save_models()
                    
                    progress_bar.progress(100)
                    st.success('Models trained and saved successfully!')
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    st.stop()

            except Exception as e:
                st.error(f"Error during initialization: {str(e)}")
                st.stop()

        # Create and display the dashboard
        try:
            analysis.create_dashboard()
        except Exception as e:
            st.error(f"Error creating dashboard: {str(e)}")
            st.stop()

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.stop()