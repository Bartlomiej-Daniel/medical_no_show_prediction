import pandas as pd
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pathlib import Path

data_dir = Path(__file__).parent.parent
data_path = data_dir / "data" / "processed"

x_test = pd.read_csv(data_path / "x_test.csv")
y_test = pd.read_csv(data_path / "y_test.csv")

model = joblib.load("modeling/model.pkl")

y_pred = model.predict(x_test)

print(classification_report(y_test, y_pred))