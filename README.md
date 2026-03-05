# Recipe Rating Predictor 🍽️

Predicts the star rating (1–5) a user will give to a recipe.

🚀 **Live App:** [reciperatingmlproject.streamlit.app](https://reciperatingmlproject-y8kciggtqj3vj48jfxia8h.streamlit.app/)

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
| Gradient Boosting | Ensemble ✅ Best |

## Results
**Best Model: Gradient Boosting**
- Scored on **F1-weighted** (handles class imbalance fairly)
- Top features: `BestScore`, `UserReputation`, `ThumbsUpCount`
- Full report in `notebooks/eda.ipynb`

## Project Structure
```
recipe_rating_ML_project/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── eda.ipynb                  # EDA 
│   └── recipe_rating_pipline.pkl  # Saved model
├── src/
│   ├── app.py                     # Streamlit web app
│   ├── preprocessing.py           # Custom pipeline & transformers
│   ├── model.py                   # Model definitions
│   └── retrain.py                 # Retrain script
└── requirements.txt
```

## How to Run Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt
```

## Retrain the Model
```bash
python src/retrain.py
```

```bash
# 2. Run the Streamlit app
python -m streamlit run src/app.py
```

