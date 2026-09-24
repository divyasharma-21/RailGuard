# RailGuard 🚆

### Predictive Maintenance & Equipment Health Intelligence System

RailGuard is a **data analytics and machine learning system** that analyzes real-world metro train compressor sensor data to detect abnormal equipment behaviour, study failure patterns, estimate maintenance risk, and provide explainable insights through an interactive dashboard.

---

## 🔍 Project Workflow

```text
MetroPT-3 Dataset
       ↓
Data Cleaning & Preprocessing
       ↓
EDA & Failure Analysis
       ↓
Feature Engineering
       ↓
Anomaly Detection
       ↓
Predictive Modelling
       ↓
Explainable AI
       ↓
Maintenance Decision Support
       ↓
Streamlit Dashboard
```

---

## 📊 Dataset

RailGuard uses the **MetroPT-3 — Metro do Porto Air Compressor Dataset** from the UCI Machine Learning Repository.

- **1,516,948** raw observations  
- **213 days** of operational data  
- Metro train Air Production Unit (APU) compressor  
- **4** documented failure events (air leak)  
- Sensor measurements including pressure, temperature, motor current, and operational signals  
- License: **CC BY 4.0**  
- DOI: [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R)  

**Official Dataset:**  
https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset  

The raw dataset is **not included** in this repository because of its large size.

---

## 🤖 Machine Learning

RailGuard uses:

- **Isolation Forest** for anomaly detection (primary component)  
- **Logistic Regression** for supervised prediction  
- **HistGradientBoosting** for supervised prediction  
- **SHAP** for model explainability  

The anomaly detection system identifies unusual equipment behaviour relative to normal-operation patterns and maps scores to **Normal / Monitor / Investigate** decision categories.

---

## 📈 Analytics & Dashboard

The Streamlit dashboard provides:

- KPI summaries  
- Sensor averages and distributions  
- Time-series analysis  
- Pressure, temperature, and motor-current analysis  
- Operational-state analysis  
- Failure-event analysis  
- Anomaly monitoring  
- Predictive model results  
- SHAP feature importance  
- Maintenance decision support  

```bash
streamlit run src/dashboard/app.py
```

---

## 🛠️ Tech Stack

- Python  
- Pandas · NumPy  
- Scikit-learn  
- Matplotlib · Plotly  
- SHAP  
- Streamlit  
- Jupyter Notebook  
- Git & GitHub  

---

## 📁 Project Structure

```text
RailGuard/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
│
├── docs/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   └── dashboard/
│
├── tests/
│
├── scripts/
│
├── RailGuard.ipynb         
├── config.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup

### Requirements

1. Python 3.10+  
2. MetroPT-3 dataset from UCI  
3. Dependencies from `requirements.txt`  

### Dataset Placement

After downloading the dataset, place these files inside `data/raw/`:

```text
MetroPT3(AirCompressor).csv
Data Description_Metro.pdf
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python scripts/run_pipeline.py
```

### Run the Dashboard

```bash
streamlit run src/dashboard/app.py
```

### Run Tests

```bash
pytest
```

---

## ⚠️ Limitations

- Only **four** documented failure events are available.  
- The dataset represents **one** compressor / system context.  
- Supervised model results are therefore statistically limited.  
- Anomaly scores indicate **unusual behaviour** and do not automatically diagnose the physical cause of failure.  
- Results are intended for **maintenance decision support**, not automated engineering decisions.  

---

## 🚀 Future Scope

- More failure events and datasets  
- Real-time sensor monitoring  
- Remaining Useful Life prediction  
- Fleet-level analysis  
- Maintenance history integration  
- Real-time alerts  

---

## 👩‍💻 Author

**Divya Sharma**  
IBM SkillsBuild — Data Analytics with AI Academic Internship Program  
BharatCares × AICTE  

---

**RailGuard** — From sensor data to explainable maintenance intelligence.
