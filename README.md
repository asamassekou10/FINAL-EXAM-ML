# Student Performance Prediction - Machine Learning Classification Project

**Course:** ITAI 1371 - Machine Learning  
**Institution:** Houston City College  
**Semester:** Fall 2025  
**Team:** Group 3

## Team Members

- **Erick Banegas** - Decision Tree, Logistic Regression, Support Vector Classifier
- **Alhassane Samassekou** - Random Forest, Gradient Boosting, Voting Ensemble, Visualizations
- **Peter Amoye** - K-Nearest Neighbors, Bayesian Ensemble, Evaluation Framework

## Project Overview

This project implements a comprehensive machine learning pipeline to predict student academic performance (Pass/Fail) using multiple classification algorithms. We achieved exceptional results with **two models reaching perfect 100% test accuracy** through careful feature engineering and proper methodology.

### Key Achievements

- **Perfect Classification:** Random Forest and Gradient Boosting both achieved 100% test accuracy
- **Robust Evaluation:** 8 models compared using 5 metrics on both validation and test sets
- **Feature Engineering:** Created 30+ engineered features with 70% appearing in top 10 predictors
- **Minimal Overfitting:** Average generalization gap of only 3.25% across all models
- **Comprehensive Analysis:** 6 professional visualizations documenting all results

## Dataset

- **Source:** Student Information Dataset
- **Size:** 1,000 student records
- **Features:** 47 total (17 original + 30 engineered)
- **Target Variable:** Binary classification (Pass/Fail)
- **Split:** 70% training, 15% validation, 15% test (stratified)

### Feature Categories

- **Demographics:** Age, gender, ethnicity, parental education
- **Academic History:** Previous GPA, SAT scores, prior failures
- **Behavioral Metrics:** Study hours, attendance rate, class participation
- **Engagement Indicators:** Assignment completion, tutoring usage, extracurricular activities
- **Engineered Features:** Score aggregations, interaction terms, polynomial features, binary indicators

## Models Implemented

### Individual Models (6 Required)

1. **Logistic Regression** - 93.3% test accuracy
2. **Decision Tree Classifier** - 99.3% test accuracy
3. **Random Forest Classifier** - 100% test accuracy ⭐
4. **Gradient Boosting Classifier** - 100% test accuracy ⭐
5. **K-Nearest Neighbors** - 88.0% test accuracy
6. **Support Vector Classifier** - 93.3% test accuracy

### Ensemble Models (2 Required)

7. **Voting Classifier** - 99.3% test accuracy (soft voting of top 3 models)
8. **Bayesian Ensemble** - 84.0% test accuracy (Gaussian Naive Bayes)

## Results Summary

### Top Performers

| Model | Validation Accuracy | Test Accuracy | Test ROC-AUC |
|-------|---------------------|---------------|--------------|
| Random Forest | 99.3% | **100.0%** | 1.000 |
| Gradient Boosting | 98.0% | **100.0%** | 1.000 |
| Decision Tree | 98.7% | 99.3% | 0.990 |
| Voting Classifier | 98.7% | 99.3% | 1.000 |

### Statistical Summary

- **Mean Test Accuracy:** 94.7%
- **Median Test Accuracy:** 96.3%
- **Standard Deviation:** 5.7%
- **Performance Range:** 84.0% - 100.0%
- **Models ≥95% Accuracy:** 4 out of 8 (50%)

### Top 10 Most Important Features

1. **total_score** (Engineered) - 16.51%
2. **avg_score** (Engineered) - 16.40%
3. **attendance_x_avg_score** (Engineered) - 14.92%
4. **avg_score_squared** (Engineered) - 13.36%
5. **high_avg_score** (Engineered) - 8.78%
6. **verbal_avg** (Engineered) - 5.37%
7. **strong_performer** (Original) - 3.59%
8. **attendance_squared** (Engineered) - 2.82%
9. **attendance_rate** (Original) - 2.75%
10. **at_risk** (Original) - 2.67%

## Repository Contents

### Core Deliverables

- **`Final_Project_Notebook_Alhassane_Samassekou_Erick_Banegas_Peter_Amoye.ipynb`**  
  Complete Jupyter notebook with all code, visualizations, and results (59 cells)

- **`student_info_dataset_Final.csv`**  
  Original dataset with 1,000 student records

- **`Final_Project_Model_Comparison_Table_Alhassane_Erick_Peter.pdf`**  
  Formatted table comparing all 8 models across 10 metrics

- **`FINAL_PROJECT_REPORT_GROUP3_Erick_Banegas_Alhassane_Samassekou_Peter_Amoye.pdf`**  
  Comprehensive 5-page analysis report with methodology, results, and discussion

### Team Documentation

- **`Final_Project_Team_Contribution_Group3_Alhassane_Erick_Peter.pdf`**  
  Combined team contributions document

- **`Final Project - Personal Contribution Report - Alhassane Samassekou.pdf`**  
  Individual contribution: Data splitting, Random Forest, Gradient Boosting, Voting Ensemble, visualizations

- **`Final Project - Personal Contribution Report - Erick Banegas.pdf`**  
  Individual contribution: Decision Tree, Logistic Regression, SVC

- **`Final Project - Personal Contribution Report - Amoye Peter.pdf`**  
  Individual contribution: KNN, Bayesian Ensemble, evaluation framework

## Technologies Used

### Programming Languages & Tools
- **Python 3.8+**
- **Jupyter Notebook**
- **Google Colab**

### Libraries & Frameworks
- **scikit-learn 1.0+** - Machine learning models and evaluation
- **pandas 1.3+** - Data manipulation and analysis
- **numpy 1.21+** - Numerical computations
- **matplotlib 3.4+** - Data visualization
- **seaborn 0.11+** - Statistical visualization

## Methodology

### 1. Data Preprocessing
- Implemented stratified 70-15-15 split (training-validation-test)
- Created preprocessing pipeline with StandardScaler and OneHotEncoder
- Prevented data leakage by fitting only on training data
- Applied feature engineering to create 30+ new features

### 2. Model Training
- Trained 6 individual classification models with optimized hyperparameters
- Developed 2 ensemble methods (Voting Classifier and Bayesian Ensemble)
- Used cross-validation for model stability assessment
- Applied class balancing where appropriate

### 3. Evaluation
- Computed 5 metrics for each model: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Evaluated on both validation and test sets (80 total metrics)
- Created 6 comprehensive visualizations documenting results
- Conducted statistical analysis of model performance

## Key Insights

### What We Learned

1. **Feature Engineering is Critical:** 70% of top predictors were engineered features, demonstrating that proper feature engineering can be more impactful than algorithm selection

2. **Academic Performance Dominates:** Total scores and average scores account for 33% of predictive power, making them the strongest indicators

3. **Engagement Matters:** The interaction between attendance and performance (attendance_x_avg_score) was the third most important feature

4. **Tree-Based Models Excel:** 5 of the top 6 performers used decision tree foundations (Random Forest, Gradient Boosting, Decision Tree, Voting Classifier)

5. **Generalization is Achievable:** With proper methodology, we achieved an average generalization gap of only 3.25%

### Practical Applications

- **Early Warning Systems:** Deploy Random Forest model for 100% accuracy in identifying at-risk students
- **Intervention Targeting:** Focus on students with low total_score and poor attendance
- **Resource Allocation:** Use top 8 features for 80% prediction power (simplified dashboard)
- **Policy Development:** Emphasize behavioral factors (study hours, attendance) over demographics

## Visualizations

The notebook includes 6 comprehensive visualizations:

1. **Confusion Matrices (2×4 Grid)** - All 8 models on test set
2. **ROC Curves (Dual Plot)** - Validation and test sets with AUC values
3. **Dual Bar Chart** - Validation vs test performance comparison
4. **Feature Importance** - Top 25 features with cumulative importance
5. **Ensemble Analysis (6-Panel)** - Comprehensive ensemble comparison
6. **Statistical Summary (6-Panel)** - Performance distribution and rankings

## How to Run

### Prerequisites
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Execution
1. Clone the repository
2. Open `Final_Project_Notebook_Alhassane_Samassekou_Erick_Banegas_Peter_Amoye.ipynb` in Jupyter or Google Colab
3. Upload `student_info_dataset_Final.csv` when prompted
4. Run all cells (Runtime → Run all)
5. Review visualizations and results

### Expected Runtime
- Total execution time: 5-10 minutes
- All 8 models train in under 5 minutes
- Visualizations generate automatically

## Model Recommendations

### For Maximum Accuracy
**Use:** Random Forest or Gradient Boosting  
**Accuracy:** 100%  
**Use Case:** Critical decisions requiring perfect classification

### For Production Deployment
**Use:** Voting Classifier  
**Accuracy:** 99.3%  
**Use Case:** Balanced robustness and performance

### For Real-Time Applications
**Use:** Bayesian Ensemble  
**Accuracy:** 84.0%  
**Use Case:** Fast inference when speed matters more than perfection

### For Interpretability
**Use:** Decision Tree  
**Accuracy:** 99.3%  
**Use Case:** Explaining predictions to non-technical stakeholders
