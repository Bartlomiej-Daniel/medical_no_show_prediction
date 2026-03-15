import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://postgres:postgres@localhost:5432/hospital_appointments"
)

df = pd.read_sql("SELECT * FROM appointments", engine)

df["scheduled_day"] = pd.to_datetime(df["scheduled_day"])
df["appointment_day"] = pd.to_datetime(df["appointment_day"])

# waiting time
df["waiting_days"] = (
    df["appointment_day"] - df["scheduled_day"]
).dt.days

df = df[df["waiting_days"] >= -1]

# weekday
df["appointment_weekday"] = df["appointment_day"].dt.weekday
df["scheduled_weekday"] = df["scheduled_day"].dt.weekday

# waiting groups
bins = [-10, -1, 3, 14, 30, 200]
labels = ["same_day", "0-3", "4-14", "15-30", "30+"]

df["waiting_group"] = pd.cut(df["waiting_days"], bins=bins, labels=labels)

df = pd.get_dummies(df, columns=["waiting_group"], drop_first=True)

df = df.drop(columns=[
    "scheduled_day",
    "appointment_day",
    "appointment_id",
    "patient_id"
])

df.to_csv("data/processed/features.csv", index=False)

print("Features dataset created")