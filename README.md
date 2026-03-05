# Recipe Rating Predictor

Predicts the star rating (1–5) a user will give to a recipe based on their comment activity.

---

## Problem Statement
Given features like user reputation, thumbs up/down, reply count, and comment timestamp, predict the recipe rating (1–5) a user will assign.

## Dataset
- **Source:** Recipes dataset (`train.csv`, `test.csv`)
- **Size:** 13,000+ rows
- **Features:** Numerical, categorical, and timestamp-based

## Approach
1. **EDA** — explored distributions, missing values, class imbalance
2. **Preprocessing** — log transform for skewed features, StandardScaler, OneHotEncoder, custom `DateFeatureExtractor` (extracts hour, month, day_of_week from Unix timestamp)
3. **Modeling** — trained multiple models inside a single sklearn `Pipeline`
4. **Tuning** — `GridSearchCV` with 5-fold CV, scored on `f1_weighted`

## Models Used
| Model | Type |
|---|---|
| Logistic Regression | Baseline |
| Decision Tree | Tree-based |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |

## Results

**Best Model: Gradient Boosting**

Gradient Boosting outperformed all other models. It works by building trees sequentially — each tree corrects the mistakes of the previous one — making it ideal for this dataset where ratings are imbalanced and patterns are non-linear.

- Scored on **F1-weighted** (handles class imbalance fairly)
- Top features: `BestScore`, `UserReputation`, `ThumbsUpCount`and etc
- Full confusion matrix and classification report in `notebooks/eda.ipynb`

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook (trains & saves the pipeline)
jupyter notebook notebooks/eda.ipynb

```