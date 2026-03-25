from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import data_preprocessing
from data_preprocessing import load_data, preprocess_data

print("Imported from:", data_preprocessing.__file__)

df = load_data("../data/raw/internet_service_churn.csv")

print("Columns after load_data:", df.columns.tolist())

X_train, X_test, y_train, y_test, scaler, imputer = preprocess_data(df)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

feature_names = df.drop("churn", axis=1).columns
importances = model.feature_importances_

feat_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature importance:")
print(feat_importance)

print("\nClassification report (default threshold=0.5):")
print(classification_report(y_test, y_pred))

y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.4
y_pred_custom = (y_proba > threshold).astype(int)

print(f"\nClassification report (custom threshold={threshold}):")
print(classification_report(y_test, y_pred_custom))

joblib.dump(model, "../models/model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(imputer, "../models/imputer.pkl")
joblib.dump(df.drop("churn", axis=1).columns.tolist(), "../models/features.pkl")

from sklearn.metrics import roc_auc_score

y_proba = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_proba)

print("ROC-AUC:", roc)