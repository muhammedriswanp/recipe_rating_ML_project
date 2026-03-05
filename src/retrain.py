import pandas as pd
import joblib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import build_preprocessor
from model import gradient_boosting, evaluate_model
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/train.csv')
print("Columns:", df.columns.tolist())
print("Shape before cleaning:", df.shape)

# ── Clean: remove Rating == 0 (no meaning) ──
df = df[df['Rating'] != 0].copy()
print("Shape after cleaning:", df.shape)
print("Rating distribution:\n", df['Rating'].value_counts())

# Split features
X = df.drop(columns=['Rating'])
y = df['Rating']

skewed_features = ['UserReputation', 'ReplyCount', 'ThumbsUpCount', 'ThumbsDownCount', 'BestScore']
normal_features = ['RecipeNumber', 'hour', 'month']
categorical_features = ['day_of_week']

# Build and train
preprocessor = build_preprocessor(skewed_features, normal_features, categorical_features)
model = gradient_boosting(preprocessor)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = evaluate_model(model, X_train, X_test, y_train, y_test)

# Save
joblib.dump(model.best_estimator_, 'notebooks/recipe_rating_pipline.pkl')
print("✅ Saved successfully!")