from cProfile import label
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from pathlib import Path

base_dir = Path(__file__).parent.parent
data_path = base_dir / "data" / "processed"
reports_dir = base_dir / "reports"

x_test = pd.read_csv(data_path / "x_test.csv")
y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

model = joblib.load("modeling/model.pkl")

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))

feature_importance = model.feature_importances_

features = x_test.columns

importance_df = pd.DataFrame({
    "feature": features,
    "importance": feature_importance
    })

importance_df = importance_df.sort_values("importance", ascending=False)

print(importance_df)

importance_df.to_csv(
    reports_dir/ "tables" / "feature_importance.csv",
    index=False
)
  
plt.figure(figsize=(10,6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.gca().invert_yaxis()

plt.title("Feature importance for prediction no-show")
plt.xlabel("Importance")
plt.savefig(reports_dir/ "figures" / "feature_importance.png", bbox_inches="tight")
plt.show()

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Show", "No-show"],
    yticklabels=["Show", "No-show"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(reports_dir/ "figures" / "confusion_matrix.png", bbox_inches="tight")
plt.show()

# roc curve
y_probs = model.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False possitive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig(reports_dir/ "figures" / "roc_curve.png", bbox_inches="tight")
plt.show()
