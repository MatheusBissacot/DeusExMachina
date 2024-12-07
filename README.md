# Deus Ex Machina - Project CS4: (De)Generative AI

## Overview
This project explores the biases inherent in generative AI models, with a focus on Stable Diffusion, and demonstrates methodologies to address these issues using imbalanced data handling techniques. The study uses the **Credit Card Fraud Detection Dataset** as a case study to evaluate the effectiveness of various balancing techniques and their impact on model performance.

The notebook systematically guides the reader through:
1. A detailed introduction to the problem.
2. Dataset exploration and feature analysis.
3. Implementation of different machine learning models (Random Forest and XGBoost).
4. Evaluation of balancing techniques like SMOTE, ADASYN, and SMOTE-Tomek.
5. Comparison of models and insights into handling biases and imbalances.

---

## Dataset
The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download), which consists of:
- **284,807 transactions** with **492 fraudulent cases (0.172%)**.
- 28 PCA-transformed features and two untransformed ones: `Time` and `Amount`.

---

## Libraries and Tools
The following libraries are used:
- **pandas** (v2.2.3) - Data manipulation
- **seaborn** & **matplotlib** - Visualization
- **sklearn** (v1.5.2) - Preprocessing, modeling, and metrics
- **xgboost** (v2.1.2) - XGBoost implementation
- **imblearn** (v0.12.4) - Balancing techniques
- **ydata-profiling** (v4.12.0) - Profiling
- **Python** (v3.11.7)

---

## Key Steps in the Notebook
1. **Dataset Overview**: 
   - Exploration of class distribution, feature properties, and basic statistics.
   - Detection and treatment of missing or duplicate values.

2. **Model Training**:
   - Baseline models: Random Forest and XGBoost without balancing techniques.
   - Incorporation of balancing methods like SMOTE, B-SMOTE, and ADASYN to handle class imbalance.

3. **Evaluation**:
   - Metrics used include **Precision**, **Recall**, **F1 Score**, **Accuracy**, and **AUPRC**.
   - Comparisons highlight how balancing techniques affect model performance.

4. **Visualizations**:
   - Performance metrics and confusion matrices are visualized for clarity.

---

## Usage
### Running the Notebook
1. Clone the repository and ensure all dependencies are installed.
2. Load the notebook using Jupyter or any compatible environment.
3. Run the cells sequentially for a guided execution.

### Input
The notebook requires the dataset `creditcard.csv` in the same directory.

### Outputs
- Comparative analysis of models and balancing techniques.
- Insights into the mitigation of bias in machine learning.# Deus Ex Machina - Project CS4: (De)Generative AI
