#IMPORT LIBRARY
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer   #FunctionTransformer → applies custom function (like log)
from sklearn.impute import SimpleImputer # Handles missing values
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer #Converts text into numeric features so ML models can understand it, Work based word importance.
import numpy as np
import pandas as pd

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["CreationTimestamp"] = pd.to_datetime(X["CreationTimestamp"], unit='s')
        X["hour"] = X["CreationTimestamp"].dt.hour
        X["day_of_week"] = X["CreationTimestamp"].dt.dayofweek
        X["month"] = X["CreationTimestamp"].dt.month
        return X

#Build Preprocessing Pipelines
def build_preprocessor(skewed_numeric_features, normal_numeric_features, categorical_features, text_feature):
    
    # Skewed Numeric Pipeline 
    skewed_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # Replace missing numeric values with median (robust to outliers)
        ("log_transform", FunctionTransformer(np.log1p)), # Apply log(1 + x) transformation to reduce skewness
        ("scaler", StandardScaler())
    ])

    # Normal Numeric Pipeline (only scale)
    normal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical Pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))    
    ])

    #Text Pipeline
    text_pipeline = Pipeline([
        ("imputer",SimpleImputer(strategy="constant",fill_value="")),
        ("flatten", FunctionTransformer(lambda x: x.ravel(), validate=False)), #.ravel() converts 2D → 1D
        ("tfidf", TfidfVectorizer(max_features=3000))   #Keep only the top 3000 most important words
    ])

    #Combine Using ColumnTransformer
    column_transformer = ColumnTransformer(transformers=[
        ("skewed_num", skewed_pipeline, skewed_numeric_features),
        ("normal_num", normal_pipeline, normal_numeric_features),
        ("cat", categorical_pipeline, categorical_features),
        ("text", text_pipeline, text_feature)
    ])

    preprocessor = Pipeline([
        ("date_features", DateFeatureExtractor()),
        ("column_transform", column_transformer)
    ])


    return preprocessor

# skewed_numeric_features = ["UserReputation", "ReplyCount", "ThumbsUpCount", "ThumbsDownCount", "BestScore"]

# normal_numeric_features = ["RecipeNumber", "hour", "month"]

# categorical_features = ["day_of_week"]

# text_feature = "Recipe_Review" # Pass as a string (not a list). TfidfVectorizer expect 1D array.