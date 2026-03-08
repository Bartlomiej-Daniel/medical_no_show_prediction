import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://postgres:postgres@localhost:5432/hospital_appointments"
    )

df = pd.read_sql("SELECT * FROM appointments", engine)


print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df['age'].mean())

# data cleaning
df = df[df["age"] >= 0]
df = df[df["age"] <= 100]

sns.set_style("whitegrid")

# Age distribution
plt.figure(figsize=(8,6))

sns.histplot(df["age"], bins=50)

plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.show()

# No-show rate by age
age_no_show = df.groupby("age")["no_show"].mean()

plt.figure(figsize=(10,6))

sns.lineplot(x=age_no_show.index, y=age_no_show.values)

plt.title("No-show rate by age")
plt.xlabel("Age")
plt.ylabel("No-show rate")

plt.show()

# No-show rate by gender
plt.figure(figsize=(6,4))

sns.barplot(data=df, x="gender", y="no_show")

plt.title("No-show rate by gender")

plt.show()

# No-show rate by SMS reminder
plt.figure(figsize=(6,4))

sns.barplot(data=df, x="sms_received", y="no_show")

plt.title("No-show rate by SMS reminder")

plt.show()
