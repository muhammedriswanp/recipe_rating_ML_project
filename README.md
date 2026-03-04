# Recipe Rating Prediction — ML Project

In this project, I built a machine learning model to predict recipe ratings using different recipe features. I worked on everything step by step — from exploring the data to training, evaluating, and saving the final model pipeline.

---

## Problem Statement

The goal of this project is to predict a recipe’s rating before it gets many reviews. I used past data like user reputation, likes, dislikes, and submission time, and treated it as a multiclass classification problem.

---

## Dataset

- **Source:** Recipes dataset (which has train.csv and test.csv)
- **Size:** Over 13000+ rows
- **Feature types:** Both numerical and categorical features
- **Preprocessing needed:** Missing values, skewed distributions, date/time columns, categorical encoding

---

## Project Structure

```
recipe_rating_ML_project/
│
├── data/
│   ├── train.csv           # Training data
│   └── test.csv            # Test data
│
├── notebooks/
│   ├── eda.ipynb           # Full EDA + model training notebook
│   └── recipe_rating_pipline.pkl   # Saved pipeline (inside notebooks folder)
│
├── src/
│   ├── preprocessing.py    # Custom transformers and preprocessor builder
│   ├── model.py            # Model definitions and evaluation function
│   └── utils.py            # Utility helpers
│
├── requirements.txt        # Python dependencies
└── README.md               # You're reading it!
```

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Checked for missing values 
- Looked at the distribution of numerical features — many were skewed
- Explored the target variable distribution (it's quite imbalanced across rating classes)
- Extracted useful time-based features from the `CreationTimestamp` column (hour, day of week, month)

### 2. Preprocessing Pipeline
All preprocessing is done **inside a build_preprocessor Pipeline**.

- **Skewed numerical features** → Median imputation → Log transform (`log1p`) → StandardScaler
- **Normal numerical features** → Median imputation → StandardScaler
- **Categorical features** → Most frequent imputation → OneHotEncoder
- **Date column** → Custom `DateFeatureExtractor` transformer (extracts hour, day_of_week, month)

Everything is combined using `ColumnTransformer` so preprocessing and modeling stay as one clean workflow.

### 3. Models Used

| Model | Type |
|---|---|
| Logistic Regression | Baseline (Linear) |
| Decision Tree | Tree-based |
| Random Forest | Ensemble (Bagging) |
| Gradient Boosting | Ensemble (Boosting) |

**Why these models?**
- Logistic Regression gives us a simple baseline to beat
- Decision Tree shows how a single tree handles this problem
- Random Forest reduces overfitting through bagging — good for noisy data
- Gradient Boosting focuses on hard-to-classify samples iteratively — usually one of the best performers

### 4. Hyperparameter Tuning
Tuning is done **inside the pipeline** using `GridSearchCV` with 5-fold cross-validation:

- **Random Forest:** `max_depth`, `min_samples_split`
- **Gradient Boosting:** `n_estimators`, `learning_rate`

Scoring metric used: `f1_weighted` — appropriate for imbalanced multiclass problems.

### 5. Evaluation Metrics
- Classification Report (Precision, Recall, F1-score)
- Accuracy
- Confusion Matrix
- ROC-AUC Score (multiclass, `ovr` strategy, weighted average)
- Cross-validation scores

### 6. Feature Importance
Feature importance was extracted from the Gradient Boosting to understand which features drive the predictions the most. The top features are discussed in the EDA.

---

## Results

All model results and comparisons are documented inside `notebooks/eda.ipynb`. The notebook covers:
- Side-by-side comparison of all models
- Best hyperparameters found by GridSearch
- Final evaluation on the test set
- Feature importance plots and interpretation

---

## How to Run

### 1. Clone the repository

```bash
git clone <https://github.com/muhammedriswanp/recipe_rating_ML_project/tree/main>
cd recipe_rating_ML_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

```bash
jupyter notebook notebooks/eda.ipynb
```

Run all cells from top to bottom. The notebook handles loading data, preprocessing, training all models, evaluating them, and saving the final pipeline.

### 4. The saved pipeline

The trained pipeline is saved at:
```
notebooks/recipe_rating_pipline.pkl
```

You can load it anytime:

```python
import joblib

pipeline = joblib.load("notebooks/recipe_rating_pipline.pkl")
predictions = pipeline.predict(new_data)
```

---