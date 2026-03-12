import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


data_dir = Path(__file__).resolve().parent.parent
data_file = data_dir / "data" / "processed" 

df = pd.read_csv(data_file / "features.csv")

features = [
    "age",
    "waiting_days",
    "sms_received",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "handicap",
    "appointment_weekday"
]

x = df[features]
y = df["no_show"]

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size=0.2, 
    random_state=42
    )

x_test.to_csv(data_file / "x_test.csv", index=False)
y_test.to_csv(data_file / "y_test.csv", index=False)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
    )

model.fit(x_train, y_train)

model_dir = Path(__file__).parent
joblib.dump(model, model_dir / "model.pkl")

print("Model trained and saved")

