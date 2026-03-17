# Predicting Medical Appointment No-Shows

## Project Overview

Missed medical appointments (no-shows) are a major problem in healthcare systems.  
They lead to inefficient use of medical staff time, longer waiting lists, and financial losses.

The goal of this project is to analyze factors influencing appointment attendance and build a machine learning model that predicts whether a patient will miss their appointment.

This project demonstrates a full **data analytics pipeline** including:

- ETL pipeline
- PostgreSQL database
- Exploratory Data Analysis
- Feature engineering
- Machine learning modeling
- Neural network experiment with PyTorch
- Business insights

---

# Dataset

The project uses the publicly available dataset:

Medical Appointment No Shows Dataset

Number of records:

110,527 appointments

Features include:

- patient demographics (age, gender)
- medical conditions
- appointment scheduling data
- SMS reminders
- neighborhood information

Target variable:

```
no_show
```

```
0 = patient attended
1 = patient missed the appointment
```

Approximately **20% of appointments result in no-shows**.

---

# Project Structure

```

project/

data/
raw/
processed/

db/
schema.sql

etl/
load_to_postgres.py

analysis/
eda.py

features/
build_features.py

modeling/
train_model.py
evaluate_model.py
train_pytorch_model.py

notebooks/
exploration.ipynb

README.md
requirements.txt

```

---

# Data Pipeline

## ETL

Raw dataset is loaded and stored in a PostgreSQL database.

Pipeline steps:

1. Load raw CSV
2. Clean columns
3. Store data in PostgreSQL
4. Query data for analysis

Tools used:

- Python
- Pandas
- PostgreSQL
- SQLAlchemy

---

# Exploratory Data Analysis

EDA was used to identify patterns related to missed appointments.

Key analyses:

- age distribution
- no-show rate by age
- no-show rate by gender
- impact of SMS reminders
- waiting time before appointment

Key observation:

Longer waiting times are associated with higher no-show rates.

---

# Feature Engineering

New features were created to improve model performance.

Examples:

**waiting_days**

```
appointment_day - scheduled_day
```

**appointment_weekday**

Day of week when appointment takes place.

**waiting_group**

Grouped waiting time categories:

```
same_day
0–3 days
4–14 days
15–30 days
30+ days
```

These features significantly improved model performance.

---

# Machine Learning Models

Two approaches were tested.

## Random Forest

A Random Forest classifier was trained using engineered features.

Model advantages:

- handles tabular data well
- robust to feature scaling
- interpretable feature importance

Key features influencing predictions:

- waiting_days
- age
- appointment_weekday

---

## Neural Network (PyTorch)

A simple neural network was implemented using PyTorch to compare deep learning with classical machine learning methods.

Architecture:

```

Input Layer
↓
16 neurons (ReLU)
↓
8 neurons (ReLU)
↓
1 output neuron (Sigmoid)

```

The neural network achieved performance similar to Random Forest.

This result suggests that **classical machine learning algorithms are well suited for structured tabular healthcare data**.

---

# Model Performance

Example evaluation metrics:

```

precision recall f1-score support

0 (show) 0.84 0.80 0.82
1 (no-show) 0.31 0.37 0.34

accuracy: ~72%

```

The dataset is imbalanced (~80% show vs 20% no-show), which makes predicting no-shows more difficult.

---

# Feature Importance

Top predictive features:

1 waiting_days  
2 age  
3 appointment_weekday  
4 sms_received

Waiting time before appointment was the strongest predictor of missed visits.

---

# Key Insights

1️ Around **20% of appointments are missed**.

2️ **Long waiting times significantly increase no-show probability.**

3️ **Same-day appointments have the lowest no-show rate.**

4️ SMS reminders have a small but measurable effect.

---

# Business Implications

Healthcare providers could reduce missed appointments by:

- minimizing waiting times
- sending appointment reminders
- prioritizing high-risk patients for follow-up
- slightly overbooking appointments with high no-show probability

Even small improvements in attendance rates could significantly increase hospital efficiency.

---

# Technologies Used

Python  
Pandas  
Matplotlib / Seaborn  
PostgreSQL  
SQLAlchemy  
Scikit-learn  
PyTorch

---

# Future Improvements

Possible extensions of this project:

- hyperparameter tuning
- cross-validation
- SHAP model interpretability
- deployment as a prediction API

---

# How to Run the Project

## 1. Install dependencies

pip install -r requirements.txt

## 2. Run the pipeline

python etl/load_to_postgres.py  
python features/build_features.py  
python modeling/train_model.py  
python modeling/evaluate_model.py  
python modeling/train_pytorch_model.py

---

# Author

Data Analytics Portfolio Project

Created to demonstrate skills in:

- data engineering
- exploratory analysis
- machine learning
- model evaluation