import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# load dataset
patients = pd.read_csv("ml_dataset.csv")

# define features/targets
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

# train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# model
model = xgb.XGBClassifier(
    n_estimators= 400,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    n_jobs=4   # important
)

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# output
print("\nXGBoost Results:")
print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"AUC:         {auc:.3f}")