    #Baseline Model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def baseline(preprocessor):
    model = Pipeline([
        ("preprocessor",preprocessor),
        ("classifier",LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    return model

#Tree Model
def decision_tree(preprocessor):
    model = Pipeline([
        ("preprocessor",preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
    return model

#Ensemble Model (Bagging)
def random_forest(preprocessor):
    model = Pipeline([
        ("preprocessor",preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [3, 5, 8]
    }

    grid  = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1_weighted",#Highly imbalanced multiclass use f1_weighted
        n_jobs=-1
    )

    return grid

#Add Boosting Model
def gradient_boosting(preprocessor):

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])

    param_grid = {
        "classifier__n_estimators": [40, 50, 75],
        "classifier__learning_rate": [0.05, 0.1, 0.2]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_weighted",#Highly imbalanced multiclass use f1_weighted
        n_jobs=-1
    )

    return grid

# Evaluation Function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    if hasattr(model, 'best_score_'):
        print(f"mean F1 of the best combination: {model.best_score_:.4f}")
        print(f"Best Params: {model.best_params_}")
    else:
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        print(f"Cross-validation(5) mean : {cv_score.mean():.4f}")

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test) 
    #One-vs-Rest means model calculates ROC AUC by comparing each class against all other classes separately.
    auc = roc_auc_score(y_test, y_probs, multi_class='ovr', average='weighted')
    # A higher AUC score indicates a better ability to distinguish between classes, with 1.0 being a perfect score and 0.5 representing random guessing

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("confusion_matrix \n")
    print(confusion_matrix(y_test, y_pred))
    print(f" Multiclass ROC AUC Score (Weighted): {auc:.4f}\n")
    
    return model
