import pandas as pd
import numpy as np

import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)

import matplotlib.pyplot as plt

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

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# xgboost model
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

# predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("\nXGBOOST RESULTS")
print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUC:         {auc:.3f}")


# SHAP INTERPRETABILITY

# create explainer
explainer = shap.Explainer(model, X_train)

# compute SHAP values
shap_values = explainer(X_test)

#show global importance
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot - Global Feature Impact")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)

#feature importance
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=300)

#single patient explanation
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.title("SHAP Explanation - Single Patient")
plt.tight_layout()
plt.savefig("shap_patient_0.png", dpi=300)

print("\nSHAP plots saved: shap_summary.png, shap_bar.png, shap_patient_0.png")