# 🏥 Diabetes Readmission Prediction

This project predicts whether a diabetic patient will be readmitted to the hospital using machine learning.

---

## 🚀 Live Demo
🔗 [https://your-app.streamlit.app](https://diabetes-readmission-ml-nvzkdxudbqfejnrc3gtsr3.streamlit.app/)

---

## 📌 Project Overview
This project builds an end-to-end machine learning pipeline to predict hospital readmission risk for diabetic patients based on clinical and hospital data.

The model helps identify high-risk patients who may need closer monitoring after discharge.

---

## ⚙️ Features
- Full ML pipeline (cleaning → encoding → feature selection → PCA → model)
- Random Forest model
- Interactive web app using Streamlit
- Real-time prediction based on user input

---

## 📊 Dataset
- Diabetes dataset from 130 US hospitals (1999–2008)
- ~100,000 patient records
- Includes demographics, medications, lab results, and hospital visits

---

## 🧠 Model
- Random Forest (best performing)
- Accuracy ~0.60
- AUC ~0.63

---

## 🔍 Key Insights
- Prior inpatient visits are the strongest predictor
- Older patients have higher readmission risk
- Hospital utilization features are more important than many medications

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

---

## 📁 Project Structure
streamlit_app.py # Main app
diabetic_data.csv # Dataset
requirements.txt # Dependencies
DS_FinalProject (1).ipynb # Analysis notebook

---

## 👩‍💻 Author
Jumanah Al-Nahdi

---

## ⚠️ Disclaimer
This project is for educational purposes only and should not be used for real medical decisions.
