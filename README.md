# 🍄 Smart Mushroom Classification

A complete Machine Learning project that classifies mushrooms as **Edible** or **Poisonous** using multiple classification algorithms.

---

## 🚀 Project Overview

This project explores different supervised machine learning algorithms for mushroom classification.

Models Used:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

The project also demonstrates:

- Data Cleaning
- One-Hot Encoding
- Feature Scaling
- Model Comparison
- Model Saving using Joblib

---

## 📂 Project Structure

```
Smart Mushroom Classification/

│
├── data/
│   └── mushrooms.csv
│
├── images/
│   ├── class_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│
├── models/
│   ├── mushroom_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── mushroom_classification.ipynb
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Dataset

Rows: **8124**

Columns: **23**

Target:

- Edible
- Poisonous

---

## 🧠 Machine Learning Workflow

- Data Analysis
- Data Cleaning
- One-Hot Encoding
- Train-Test Split
- Feature Scaling
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- Model Evaluation
- Model Comparison

---

## 📈 Model Performance

| Model | Accuracy |
|--------|---------:|
| Logistic Regression | 100% |
| Decision Tree | 100% |
| Random Forest | 100% |
| Support Vector Machine | 100% |

---

## 🛠 Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## ▶️ Run

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

---

## 👨‍💻 Author 
  Jaskaran Singh

"This application is designed to demonstrate the complete machine learning workflow using the UCI Mushroom dataset. Because the dataset contains expert-level botanical features, the interface reflects those attributes and is intended for educational purposes rather than real-world mushroom identification."
