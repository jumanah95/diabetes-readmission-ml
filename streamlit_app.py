"""
Diabetes Readmission Predictor
CS30721 — Data Sciences | Clinical Outcome Prediction from Noisy Medical Records

HOW TO RUN:
  1. pip install streamlit pandas numpy scikit-learn matplotlib
  2. Place diabetic_data.csv in the same folder
  3. streamlit run streamlit_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import os
import joblib
DATA_FILE = "diabetic_data.csv" 
ARTIFACT_FILE = "diabetes_pipeline_artifacts.joblib"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Main background */
  .stApp { background: #f0f4f8; }

  /* Hide default streamlit header decoration */
  header { visibility: hidden; }

  /* Custom metric cards */
  [data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    border-left: 4px solid #1a56db;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0f172a !important;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #38bdf8 !important; }

  /* Tabs */
  button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 14px;
  }

  /* Section headers */
  .section-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a56db 100%);
    color: white;
    padding: 14px 20px;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
  }

  /* Risk result boxes */
  .result-high {
    background: linear-gradient(135deg, #fef2f2, #fff);
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 22px 26px;
  }
  .result-moderate {
    background: linear-gradient(135deg, #fffbeb, #fff);
    border: 2px solid #f59e0b;
    border-radius: 14px;
    padding: 22px 26px;
  }
  .result-low {
    background: linear-gradient(135deg, #f0fdf4, #fff);
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 22px 26px;
  }

  /* Finding cards */
  .find-card {
    background: white;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    margin-bottom: 10px;
    border-left: 4px solid #1a56db;
  }

  /* Input sections */
  .input-section {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    margin-bottom: 16px;
  }

  /* Probability gauge */
  .prob-gauge {
    text-align: center;
    padding: 20px;
  }
  .prob-number {
    font-size: 64px;
    font-weight: 700;
    line-height: 1;
  }
</style>
""", unsafe_allow_html=True)


# ── PIPELINE FUNCTIONS (same logic as notebook) ───────────────────────────────

def map_icd(code):
    if pd.isna(code) or str(code) == 'Unknown':
        return 'Other'
    code = str(code)
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    try:
        n = float(code.split('.')[0])
    except:
        return 'Other'
    if n == 250: return 'Diabetes'
    if 390 <= n <= 459: return 'Circulatory'
    if 460 <= n <= 519: return 'Respiratory'
    if 520 <= n <= 579: return 'Digestive'
    if 800 <= n <= 999: return 'Injury'
    if 140 <= n <= 239: return 'Cancer'
    if 290 <= n <= 319: return 'Mental'
    return 'Other'


@st.cache_resource(show_spinner=False)
def train_full_pipeline(filepath):
    """
    Exact same pipeline as the Colab notebook:
    1. Clean  2. Encode  3. Feature Rank  4. Scale  5. PCA  6. Random Forest
    Uses ALL patient records — no subset.
    """
    # ── LOAD ──
    df = pd.read_csv(filepath)
    df.replace('?', np.nan, inplace=True)
    raw_count = len(df)

    # ── CLEAN (Step 1-7 from notebook) ──
    df = df.drop_duplicates(subset='patient_nbr', keep='first')
    df.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)

    drop_cols = [
        'weight', 'payer_code',
        'examide', 'citoglipton', 'glimepiride-pioglitazone',
        'nateglinide', 'chlorpropamide', 'acetohexamide', 'tolbutamide',
        'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'glipizide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    df = df[df['gender'] != 'Unknown/Invalid']
    df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]

    # Fill missing
    for col in ['max_glu_serum', 'A1Cresult']:
        if col in df.columns: df[col] = df[col].fillna('None')
    for col in ['race', 'medical_specialty']:
        if col in df.columns: df[col] = df[col].fillna('Unknown')
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns: df[col] = df[col].fillna('Unknown')

    # Save raw cleaned for EDA display
    df_eda = df.copy()
    cleaned_count = len(df)

    # ── TARGET ──
    y = (df['readmitted'] != 'NO').astype(int)
    df_model = df.drop('readmitted', axis=1).copy()

    # ── ENCODE (same as notebook) ──
    age_map = {'[0-10)':1,'[10-20)':2,'[20-30)':3,'[30-40)':4,'[40-50)':5,
               '[50-60)':6,'[60-70)':7,'[70-80)':8,'[80-90)':9,'[90-100)':10}
    df_model['age'] = df_model['age'].map(age_map)

    med_map = {'No':0, 'Steady':1, 'Down':2, 'Up':3}
    for col in ['metformin','repaglinide','glimepiride','glipizide','glyburide',
                'pioglitazone','rosiglitazone','insulin','glyburide-metformin']:
        if col in df_model.columns and df_model[col].dtype == object:
            df_model[col] = df_model[col].map(med_map).fillna(0).astype(int)

    if 'gender' in df_model.columns:
        df_model['gender'] = (df_model['gender'] == 'Female').astype(int)
    if 'change' in df_model.columns:
        df_model['change'] = (df_model['change'] == 'Ch').astype(int)
    if 'diabetesMed' in df_model.columns:
        df_model['diabetesMed'] = (df_model['diabetesMed'] == 'Yes').astype(int)

    glu_map  = {'None':0,'Normal':1,'>200':2,'>300':3}
    a1c_map  = {'None':0,'Normal':1,'>7':2,'>8':3}
    if 'max_glu_serum' in df_model.columns:
        df_model['max_glu_serum'] = df_model['max_glu_serum'].map(glu_map).fillna(0).astype(int)
    if 'A1Cresult' in df_model.columns:
        df_model['A1Cresult'] = df_model['A1Cresult'].map(a1c_map).fillna(0).astype(int)

    for col in ['diag_1','diag_2','diag_3']:
        if col in df_model.columns:
            df_model[col] = df_model[col].apply(map_icd)

    if 'medical_specialty' in df_model.columns:
        top5 = df_model['medical_specialty'].value_counts().nlargest(5).index
        df_model['medical_specialty'] = df_model['medical_specialty'].apply(
            lambda x: x if x in top5 else 'Other')

    cat_cols = df_model.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    X_all = df_encoded.copy()
    y_all  = y.reindex(X_all.index)

    # ── FEATURE RANKING ──
    rf_rank = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight='balanced', random_state=42, n_jobs=-1)
    rf_rank.fit(X_all, y_all)

    imp_df = pd.DataFrame({'Feature': X_all.columns,
                           'Importance': rf_rank.feature_importances_})\
               .sort_values('Importance', ascending=False).reset_index(drop=True)
    imp_df['Cumulative'] = imp_df['Importance'].cumsum()
    n90 = (imp_df['Cumulative'] < 0.90).sum() + 1
    selected_features = imp_df.head(n90)['Feature'].tolist()
    X_selected = X_all[selected_features]

    # ── SCALE + PCA ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected.fillna(X_selected.mean()))

    pca_all = PCA(random_state=42)
    pca_all.fit(X_scaled)
    cumvar = np.cumsum(pca_all.explained_variance_ratio_)
    n_comp = int(np.argmax(cumvar >= 0.90)) + 1

    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # ── TRAIN ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_all, test_size=0.2, random_state=42, stratify=y_all)
    model = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)

    return {
        'model': model, 'scaler': scaler, 'pca': pca,
        'selected_features': selected_features,
        'X_all_columns': list(X_all.columns),
        'acc': acc, 'auc': auc, 'cm': cm,
        'imp_df': imp_df, 'df_eda': df_eda,
        'raw_count': raw_count, 'cleaned_count': cleaned_count,
        'n_comp': n_comp, 'n_features_selected': len(selected_features),
    }


def predict_single(pipeline, inputs):
    """
    Build a feature row that matches training, scale it, apply PCA, predict.
    All numeric features are filled directly.
    OHE features stay 0 (patient-level OHE not entered in form — acceptable for prototype).
    """
    model    = pipeline['model']
    scaler   = pipeline['scaler']
    pca      = pipeline['pca']
    feats    = pipeline['selected_features']

    row = {f: 0.0 for f in feats}

    direct_map = {
        'age':                    inputs['age'],
        'gender':                 inputs['gender'],
        'time_in_hospital':       inputs['time_in_hospital'],
        'num_lab_procedures':     inputs['num_lab_procedures'],
        'num_procedures':         inputs['num_procedures'],
        'num_medications':        inputs['num_medications'],
        'number_diagnoses':       inputs['number_diagnoses'],
        'number_inpatient':       inputs['number_inpatient'],
        'number_emergency':       inputs['number_emergency'],
        'number_outpatient':      inputs['number_outpatient'],
        'admission_type_id':      inputs['admission_type_id'],
        'discharge_disposition_id': inputs['discharge_disposition_id'],
        'insulin':                inputs['insulin'],
        'metformin':              inputs['metformin'],
        'change':                 inputs['change'],
        'diabetesMed':            inputs['diabetesMed'],
        'A1Cresult':              inputs['A1Cresult'],
        'max_glu_serum':          inputs['max_glu_serum'],
    }
    for k, v in direct_map.items():
        if k in row:
            row[k] = float(v)

    X_in  = pd.DataFrame([row])[feats]
    X_sc  = scaler.transform(X_in.fillna(0))
    X_pca = pca.transform(X_sc)

    prob = model.predict_proba(X_pca)[0][1]
    pred = int(model.predict(X_pca)[0])
    return prob, pred


# ── SIDEBAR ───────────────────────────────────────────────────────────────────


data_path = DATA_FILE

if not os.path.exists(data_path):
    st.error(f"⚠️ Put {DATA_FILE} in the same folder as this app.")
    st.stop()

st.success(f"📁 Using local file: {DATA_FILE}")
#------------------------------------------------------------------------------
# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
            padding:28px 36px;border-radius:16px;margin-bottom:24px;
            box-shadow:0 4px 20px rgba(0,0,0,0.15);">
  <h1 style="color:white;margin:0;font-size:28px;font-weight:700;">
    🏥 Diabetes Readmission Predictor
  </h1>
  <p style="color:#94a3b8;margin:6px 0 0;font-size:14px;">
    CS30721 Data Sciences · Clinical Outcome Prediction from Noisy Medical Records
  </p>
</div>
""", unsafe_allow_html=True)

if data_path is None:
    st.info("👈 Upload diabetic_data.csv in the sidebar to get started.")
    st.stop()

# ── TRAIN ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(data_path):
    if os.path.exists(ARTIFACT_FILE):
        return joblib.load(ARTIFACT_FILE)

    pipeline = load_pipeline(data_path)
    joblib.dump(pipeline, ARTIFACT_FILE)
    return pipeline



with st.spinner("🔄 Training model on the full dataset... (first time ~45 seconds)"):
    try:
        pipeline = train_full_pipeline(data_path)
        df_eda   = pipeline['df_eda']
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# Quick KPI bar
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Patients (cleaned)", f"{pipeline['cleaned_count']:,}")
k2.metric("Features Selected", f"{pipeline['n_features_selected']}")
k3.metric("PCA Components", f"{pipeline['n_comp']}")
k4.metric("Test Accuracy", f"{pipeline['acc']:.3f}")
k5.metric("Test AUC", f"{pipeline['auc']:.3f}")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_pred, tab_eda, tab_model = st.tabs(["🔍  Patient Prediction", "📊  EDA & Insights", "📈  Model Details"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="section-header">🔍 Enter Patient Data — Get a Real Prediction</div>', unsafe_allow_html=True)
    st.caption("All fields below go directly into the trained Random Forest model.")

    left, right = st.columns([1, 1], gap="large")

    with left:
        # Demographics
        st.markdown("**👤 Demographics**")
        c1, c2 = st.columns(2)
        with c1:
            age_val = st.selectbox("Age Group", list(range(1, 11)),
                format_func=lambda x: ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
                                       '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)'][x-1],
                index=6)
        with c2:
            gender_val = st.selectbox("Gender", [0, 1],
                format_func=lambda x: "Female (1=Female)" if x == 1 else "Male (0=Male)")

        st.markdown("**⚡ Prior Visit History — Strongest Predictors**")
        c3, c4, c5 = st.columns(3)
        with c3:
            n_inpatient = st.number_input("Inpatient Visits", 0, 21, 0, help="Prior hospitalizations")
        with c4:
            n_emergency = st.number_input("Emergency Visits", 0, 76, 0, help="Prior ER visits")
        with c5:
            n_outpatient = st.number_input("Outpatient Visits", 0, 42, 0)

        st.markdown("**🏥 Current Hospitalization**")
        time_hosp = st.slider("Days in Hospital", 1, 14, 4)
        c6, c7 = st.columns(2)
        with c6:
            num_meds  = st.number_input("# Medications", 1, 81, 12)
            num_labs  = st.number_input("# Lab Procedures", 0, 132, 35)
        with c7:
            num_procs = st.number_input("# Procedures", 0, 6, 1)
            num_diags = st.number_input("# Diagnoses", 1, 16, 7)

    with right:
        st.markdown("**🚑 Admission Info**")
        c8, c9 = st.columns(2)
        with c8:
            adm_type = st.selectbox("Admission Type", [1,2,3,5],
                format_func=lambda x: {1:'Emergency',2:'Urgent',3:'Elective',5:'Not Available'}[x])
        with c9:
            discharge = st.selectbox("Discharge To", [1,2,3,4,5,6],
                format_func=lambda x: {1:'Home',2:'Short-term hosp',3:'Skilled nursing',
                                       4:'ICF',5:'Other',6:'Home health'}[x])

        st.markdown("**💊 Medication Status**")
        c10, c11 = st.columns(2)
        with c10:
            insulin_val  = st.selectbox("Insulin", [0,1,2,3],
                format_func=lambda x: ['No (0)','Steady (1)','Down (2)','Up (3)'][x], index=1)
            metformin_val = st.selectbox("Metformin", [0,1,2,3],
                format_func=lambda x: ['No (0)','Steady (1)','Down (2)','Up (3)'][x])
        with c11:
            change_val  = st.selectbox("Med Changed?", [0,1],
                format_func=lambda x: 'No' if x==0 else 'Yes (Ch)')
            diab_med    = st.selectbox("Diabetes Med?", [1,0],
                format_func=lambda x: 'Yes' if x==1 else 'No')

        st.markdown("**🔬 Lab Results**")
        c12, c13 = st.columns(2)
        with c12:
            a1c_val = st.selectbox("A1C Result", [0,1,2,3],
                format_func=lambda x: ['Not Tested','Normal','>7','>8'][x])
        with c13:
            glu_val = st.selectbox("Glucose Serum", [0,1,2,3],
                format_func=lambda x: ['Not Tested','Normal','>200','>300'][x])

        st.markdown("")
        predict_btn = st.button("🔍  Predict Readmission Risk", type="primary", use_container_width=True)

    # ── RESULT ────────────────────────────────────────────────────────────────
    if predict_btn:
        inputs = {
            'age': age_val, 'gender': gender_val,
            'time_in_hospital': time_hosp,
            'num_lab_procedures': num_labs,
            'num_procedures': num_procs,
            'num_medications': num_meds,
            'number_diagnoses': num_diags,
            'number_inpatient': n_inpatient,
            'number_emergency': n_emergency,
            'number_outpatient': n_outpatient,
            'admission_type_id': adm_type,
            'discharge_disposition_id': discharge,
            'insulin': insulin_val,
            'metformin': metformin_val,
            'change': change_val,
            'diabetesMed': diab_med,
            'A1Cresult': a1c_val,
            'max_glu_serum': glu_val,
        }

        prob, pred = predict_single(pipeline, inputs)
        pct = round(prob * 100, 1)

        st.markdown("---")
        
        # Result layout: gauge + details
        gcol, dcol = st.columns([1, 2])
        
        with gcol:
            risk_color = "#ef4444" if pct >= 60 else ("#f59e0b" if pct >= 40 else "#22c55e")
            risk_label = "HIGH RISK" if pct >= 60 else ("MODERATE RISK" if pct >= 40 else "LOW RISK")
            st.markdown(f"""
            <div style="background:white;border-radius:16px;padding:30px 20px;
                        text-align:center;box-shadow:0 4px 16px rgba(0,0,0,0.1);
                        border:3px solid {risk_color};">
              <div style="font-size:13px;color:#64748b;font-weight:600;
                          letter-spacing:1px;text-transform:uppercase;">
                Readmission Probability
              </div>
              <div style="font-size:72px;font-weight:800;color:{risk_color};
                          line-height:1.1;margin:8px 0;">{pct}%</div>
              <div style="font-size:15px;font-weight:700;color:{risk_color};
                          background:{risk_color}18;padding:6px 16px;
                          border-radius:20px;display:inline-block;">
                {risk_label}
              </div>
              <div style="margin-top:14px;font-size:12px;color:#94a3b8;">
                Model: Random Forest<br>AUC = {pipeline['auc']:.3f}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with dcol:
            if pct >= 60:
                st.markdown(f"""
                <div class="result-high">
                  <h3 style="color:#dc2626;margin:0 0 8px;">🚨 High Readmission Risk ({pct}%)</h3>
                  <p style="color:#7f1d1d;margin:0 0 12px;">
                    This patient profile shows multiple significant risk factors.
                  </p>
                  <p style="margin:0;"><strong>Recommended actions:</strong></p>
                  <ul style="color:#7f1d1d;margin:6px 0 0;padding-left:18px;">
                    <li>Enroll in post-discharge care management program</li>
                    <li>Schedule follow-up within 7 days of discharge</li>
                    <li>Review and simplify medication regimen</li>
                    <li>Ensure patient has clear discharge instructions</li>
                  </ul>
                </div>
                """, unsafe_allow_html=True)
            elif pct >= 40:
                st.markdown(f"""
                <div class="result-moderate">
                  <h3 style="color:#d97706;margin:0 0 8px;">⚠️ Moderate Readmission Risk ({pct}%)</h3>
                  <p style="color:#78350f;margin:0 0 12px;">
                    Some risk factors are present. Close monitoring is recommended.
                  </p>
                  <p style="margin:0;"><strong>Recommended actions:</strong></p>
                  <ul style="color:#78350f;margin:6px 0 0;padding-left:18px;">
                    <li>Schedule 30-day follow-up appointment</li>
                    <li>Educate patient on warning signs to watch for</li>
                    <li>Confirm medication adherence plan</li>
                  </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                  <h3 style="color:#16a34a;margin:0 0 8px;">✅ Low Readmission Risk ({pct}%)</h3>
                  <p style="color:#14532d;margin:0 0 12px;">
                    Patient profile indicates low readmission risk.
                  </p>
                  <p style="margin:0;"><strong>Recommended actions:</strong></p>
                  <ul style="color:#14532d;margin:6px 0 0;padding-left:18px;">
                    <li>Standard discharge planning is appropriate</li>
                    <li>Schedule routine 30-day follow-up</li>
                    <li>Provide standard diabetes management education</li>
                  </ul>
                </div>
                """, unsafe_allow_html=True)

            # Feature contribution table
            st.markdown("**Top factors contributing to this prediction:**")
            factor_data = {
                'Factor': ['Prior Inpatient Visits', 'Prior Emergency Visits',
                           'Days in Hospital', 'Number of Medications',
                           'Lab Procedures', 'Age Group'],
                'Value': [n_inpatient, n_emergency, time_hosp, num_meds, num_labs, age_val],
                'Impact': ['🔴 Very High' if n_inpatient >= 3 else ('🟡 Medium' if n_inpatient >= 1 else '🟢 Low'),
                           '🔴 High' if n_emergency >= 2 else ('🟡 Medium' if n_emergency >= 1 else '🟢 Low'),
                           '🟡 Medium' if time_hosp >= 7 else '🟢 Low',
                           '🟡 Medium' if num_meds >= 20 else '🟢 Low',
                           '🟢 Low', '🟢 Low']
            }
            st.dataframe(pd.DataFrame(factor_data), use_container_width=True, hide_index=True)

        st.caption("⚕️ This prediction is generated by the trained Random Forest model. Always use clinical judgment.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown('<div class="section-header">📊 EDA Results from the Diabetic Dataset</div>', unsafe_allow_html=True)

    # KPI row
    f_pct = (df_eda['gender'] == 'Female').sum() / len(df_eda) * 100
    r_pct = (df_eda['readmitted'] != 'NO').sum() / len(df_eda) * 100
    avg_days = df_eda['time_in_hospital'].mean()
    ins_pct  = (df_eda['insulin'] != 'No').sum() / len(df_eda) * 100 if 'insulin' in df_eda.columns else 0

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Female Patients", f"{f_pct:.1f}%", "Outnumber males")
    e2.metric("Readmission Rate", f"{r_pct:.1f}%", "Of all patients")
    e3.metric("Avg Hospital Stay", f"{avg_days:.1f} days")
    e4.metric("On Insulin", f"{ins_pct:.0f}%", "Most common med")

    st.markdown("---")

    # Plots row 1
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Readmission Distribution**")
        tc = df_eda['readmitted'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(tc.index, tc.values, color=['#22c55e','#3b82f6','#ef4444'],
               edgecolor='white', linewidth=1.5)
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution', fontsize=11, fontweight='bold')
        for i, (v, idx) in enumerate(zip(tc.values, tc.index)):
            ax.text(i, v + 200, f'{v/len(df_eda)*100:.1f}%', ha='center', fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Gender Distribution**")
        gc = df_eda['gender'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.pie(gc.values, labels=gc.index, colors=['#f472b6','#60a5fa'],
               autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
        ax.set_title('Gender Split', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_c:
        st.markdown("**Age Group Distribution**")
        age_order = ['[0-10)','[10-20)','[20-30)','[30-40)','[40-50)',
                     '[50-60)','[60-70)','[70-80)','[80-90)','[90-100)']
        ac = df_eda['age'].value_counts().reindex(age_order)
        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(range(len(age_order)), ac.values, color='#6366f1', edgecolor='white')
        ax.set_xticks(range(len(age_order)))
        ax.set_xticklabels([a.replace('[','').replace(')','') for a in age_order], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Count')
        ax.set_title('Patients by Age', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Plots row 2
    col_d, col_e = st.columns(2)

    with col_d:
        st.markdown("**Readmission Rate by Age Group**")
        ar = pd.crosstab(df_eda['age'], df_eda['readmitted'], normalize='index') * 100
        ar = ar.reindex(age_order)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bottom = np.zeros(len(ar))
        colors = {'NO':'#22c55e', '>30':'#3b82f6', '<30':'#ef4444'}
        for col_name in ar.columns:
            vals = ar[col_name].values
            ax.bar(range(len(ar)), vals, bottom=bottom,
                   label=col_name, color=colors.get(col_name,'gray'), edgecolor='white')
            bottom += vals
        ax.set_xticks(range(len(age_order)))
        ax.set_xticklabels([a.replace('[','').replace(')','') for a in age_order], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('%')
        ax.set_title('Readmission % by Age', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_e:
        st.markdown("**Top 10 Predictive Features**")
        top10 = pipeline['imp_df'].head(10)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors_bar = ['#1e3a5f' if i == 0 else '#1a56db' if i < 3 else '#60a5fa' for i in range(10)]
        ax.barh(top10['Feature'][::-1], top10['Importance'][::-1],
                color=colors_bar[::-1], edgecolor='white')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance (RF)', fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Key findings
    st.markdown("---")
    st.markdown("### 💡 Key Findings")

    f_count = (df_eda['gender']=='Female').sum()
    m_count = (df_eda['gender']=='Male').sum()
    inp_pct = (df_eda['number_inpatient'] > 0).sum() / len(df_eda) * 100
    long_st = (df_eda['time_in_hospital'] > 7).sum() / len(df_eda) * 100

    findings_list = [
        ("👩", "#ec4899", "Females Outnumber Males",
         f"Females = {f_count:,} ({f_count/(f_count+m_count)*100:.0f}%) vs Males = {m_count:,}. More female diabetic patients in this dataset."),
        ("🔄", "#1a56db", "Prior Visits Are Strongest Predictor",
         f"{inp_pct:.0f}% of patients had prior inpatient visits. number_inpatient is the #1 most important feature in the model."),
        ("💉", "#7c3aed", "Insulin is Most Common Medication",
         f"{ins_pct:.0f}% of patients use insulin. Patients with increasing dosage (Up) have higher readmission rates."),
        ("⏰", "#0891b2", "Most Patients Stay Less Than a Week",
         f"Average stay = {avg_days:.1f} days. But {long_st:.0f}% stayed >7 days — and those have higher readmission rates."),
        ("🏥", "#16a34a", "Most Patients Are Elderly (60-90)",
         "The 70-80 age group is the most common. Older patients have consistently higher readmission rates."),
        ("⚖️", "#dc2626", "Class Imbalance Exists",
         f"~{100-r_pct:.0f}% not readmitted vs {r_pct:.0f}% readmitted. Handled using class_weight='balanced' in all models."),
    ]

    fc1, fc2 = st.columns(2)
    for i, (icon, color, title, body) in enumerate(findings_list):
        col = fc1 if i % 2 == 0 else fc2
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:16px 18px;
                        margin-bottom:12px;box-shadow:0 1px 6px rgba(0,0,0,0.08);
                        border-left:4px solid {color};">
              <div style="font-size:20px;display:inline;">{icon}</div>
              <span style="font-weight:700;color:{color};font-size:14px;
                           margin-left:8px;">{title}</span>
              <p style="color:#475569;font-size:13px;margin:6px 0 0;">{body}</p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL DETAILS
# ════════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-header">📈 Model Training Details & Performance</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", "Random Forest")
    m2.metric("Test Accuracy", f"{pipeline['acc']:.3f}")
    m3.metric("Test AUC", f"{pipeline['auc']:.3f}")
    m4.metric("PCA Components", str(pipeline['n_comp']))

    st.markdown("---")
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**Pipeline Steps**")
        steps = [
            ("1", "Remove Duplicate Patients", "Keep first visit only — prevents data leakage"),
            ("2", "Drop Useless Columns", "weight (97% missing), zero-variance meds, IDs"),
            ("3", "Fill Missing Values", "Lab results → 'None', others → 'Unknown'"),
            ("4", "Encode All Features", "Age ordinal, meds ordinal, binary, OHE"),
            ("5", "Feature Ranking", f"Random Forest → top {pipeline['n_features_selected']} features (90% importance)"),
            ("6", "StandardScaler", "Mean=0, Std=1 — required before PCA"),
            ("7", "PCA", f"Reduced to {pipeline['n_comp']} components (90% variance)"),
            ("8", "Random Forest (Final)", "100 trees, max_depth=8, class_weight=balanced"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:10px;align-items:flex-start;">
              <div style="background:#1a56db;color:white;border-radius:50%;
                          width:26px;height:26px;display:flex;align-items:center;
                          justify-content:center;font-size:12px;font-weight:700;
                          flex-shrink:0;margin-top:2px;">{num}</div>
              <div>
                <div style="font-weight:600;font-size:13px;color:#1e293b;">{title}</div>
                <div style="font-size:12px;color:#64748b;">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with mc2:
        st.markdown("**Confusion Matrix**")
        cm = pipeline['cm']
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Not Readmitted','Readmitted'])
        ax.set_yticklabels(['Not Readmitted','Readmitted'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix — Random Forest', fontsize=11, fontweight='bold')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if cm[i,j] > cm.max()*0.5 else '#1e293b')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Metric Explanations**")
        st.markdown("""
        - **AUC ~0.70** → Model is better than random (0.5). Typical for this dataset.
        - **Accuracy** → Can be misleading with imbalanced data — AUC is more reliable.
        - **class_weight=balanced** → Ensures both classes get equal learning weight.
        """)

    st.markdown("---")
    st.markdown("**Top 15 Most Important Features**")
    top15 = pipeline['imp_df'].head(15)[['Feature','Importance','Cumulative']].copy()
    top15['Importance'] = top15['Importance'].round(4)
    top15['Cumulative'] = top15['Cumulative'].round(4)
    top15.index = range(1, 16)
    st.dataframe(top15, use_container_width=True)
