import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

patients = pd.read_csv("ml_dataset.csv")



features = [
    'anchor_age',
    'male',
    'DM',
    'HTN',
    'HF',
    'HLD',
    'CKD'
]

X = patients[features]
y = patients['MACE']



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    n_jobs=4
)

# train model
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]



accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("\nXGBoost Results:")
print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUC:         {auc:.3f}")

# risk stratification

def risk_category(prob):
    if prob < 0.10:
        return "Low Risk"
    elif prob < 0.30:
        return "Intermediate Risk"
    else:
        return "High Risk"

risk_df = X_test.copy()

risk_df['Actual_MACE'] = y_test.values
risk_df['Predicted_Probability'] = y_prob
risk_df['Risk_Category'] = risk_df['Predicted_Probability'].apply(risk_category)

# view results

print("\nRisk Stratification Preview:")
print(risk_df.head())

# save
risk_df.to_csv("risk_stratification_results.csv", index=False)