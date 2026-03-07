import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

# load CSV
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "raw" / "medical_appointments.csv"

df = pd.read_csv(data_path)

# rename columns to match SQL table
df = df.rename(columns={
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "SMS_received": "sms_received",
    "No-show": "no_show"
})

# clean column names
df.columns = df.columns.str.lower().str.replace("-","_")

# convert target variables
df["no_show"] = df["no_show"].map({
    "Yes": 1,
    "No": 0
    })

# create connections to PostgreSQL
engine = create_engine(
    "postgresql://postgres:postgres@localhost:5432/hospital_appointments"
    )

# load data into database
df.to_sql(
    "appointments",
    engine,
    if_exists="append",
    index=False
    )


print("Data successfully loaded to PostgreSQL!")