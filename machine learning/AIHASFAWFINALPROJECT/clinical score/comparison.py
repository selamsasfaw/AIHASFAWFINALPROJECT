import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)

import xgboost as xgb

# load data
patients = pd.read_csv("ml_dataset.csv")

# features
features = [
    "anchor_age",
    "male",
    "DM",
    "HTN",
    "HF",
    "HLD",
    "CKD"
]

X = patients[features]
y = patients["MACE"]

# train/split for xgboost
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#the xgboost model + parameters
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=4
)

model.fit(X_train, y_train)

y_pred_xgb = model.predict(X_test)
y_prob_xgb = model.predict_proba(X_test)[:, 1]

# xgboost metrics
acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb)
rec_xgb = recall_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
spec_xgb = tn / (tn + fp)

# clinical heart score

# age score
def age_score(age):
    if age < 50:
        return 0
    elif age < 65:
        return 1
    else:
        return 2

patients["age_s"] = patients["anchor_age"].apply(age_score)

# risk score creation (with same variables as XGBoost)
patients["risk_count"] = (
    patients["male"] +
    patients["DM"] +
    patients["HTN"] +
    patients["HF"] +
    patients["HLD"] +
    patients["CKD"]
)

def risk_score(x):
    if x <= 2:
        return 0
    elif x <= 4:
        return 1
    else:
        return 2

patients["risk_s"] = patients["risk_count"].apply(risk_score)

# add up variables 
patients["clinical_score"] = (
    patients["age_s"] +
    patients["risk_s"]
)

# convert to probability
patients["prob"] = patients["clinical_score"] / 5

# Clinical metrics
y_true = patients["MACE"]
y_prob = patients["prob"]
y_pred_clin = (y_prob > 0.5).astype(int)

acc_clin = accuracy_score(y_true, y_pred_clin)
prec_clin = precision_score(y_true, y_pred_clin)
rec_clin = recall_score(y_true, y_pred_clin)
auc_clin = roc_auc_score(y_true, y_prob)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_clin).ravel()
spec_clin = tn / (tn + fp)

# print results for comparison
print("XGBOOST RESULTS")
print(f"Accuracy:    {acc_xgb:.3f}")
print(f"Precision:   {prec_xgb:.3f}")
print(f"Recall:      {rec_xgb:.3f}")
print(f"Specificity: {spec_xgb:.3f}")
print(f"AUC:         {auc_xgb:.3f}")

print("CLINICAL SCORE RESULTS")
print(f"Accuracy:    {acc_clin:.3f}")
print(f"Precision:   {prec_clin:.3f}")
print(f"Recall:      {rec_clin:.3f}")
print(f"Specificity: {spec_clin:.3f}")
print(f"AUC:         {auc_clin:.3f}")