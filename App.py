import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import requests
from streamlit_lottie import st_lottie
import json
import os

# ---------------------- Page Configuration ----------------------
st.set_page_config(
    page_title="FinShield AI | Corporate Bankruptcy Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Custom CSS & Glassmorphism Theme ----------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.3) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }

        .auth-container {
            max-width: 480px;
            margin: 40px auto;
            padding: 40px;
            background: #ffffff;
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
            border: 1px solid #e2e8f0;
            text-align: center;
        }

        .auth-header {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.5rem;
        }

        .auth-subtitle {
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }

        .prediction-card {
            padding: 28px;
            border-radius: 20px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            margin: 15px 0;
        }

        .prediction-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.2);
        }

        .success-gradient {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        }

        .danger-gradient {
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            animation: pulse-danger 2.5s infinite;
        }

        @keyframes pulse-danger {
            0%, 100% { box-shadow: 0 10px 25px -5px rgba(220, 38, 38, 0.4); }
            50% { box-shadow: 0 20px 35px 5px rgba(239, 68, 68, 0.6); }
        }

        .stat-card {
            background: white;
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease;
        }

        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        }

        .user-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: #f1f5f9;
            padding: 8px 16px;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #334155;
            border: 1px solid #cbd5e1;
            margin-bottom: 15px;
        }

        .risk-pill {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .risk-pill-high { background: #fee2e2; color: #991b1b; }
        .risk-pill-medium { background: #fef3c7; color: #92400e; }
        .risk-pill-low { background: #dcfce7; color: #166534; }

        .tier-badge {
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -1px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Session State & History ----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------- Helper: Lottie Loader ----------------------
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

lottie_financial = load_lottieurl("https://lottie.host/85a2d61b-9e42-4f0e-b016-8656f4d3261a/8pQZkYJmIq.json")

# ---------------------- Load & Preprocess Data ----------------------
@st.cache_data
def load_data(file_source=None):
    if file_source is None:
        if os.path.exists("bankruptcy-prevention.xlsx"):
            file_source = "bankruptcy-prevention.xlsx"
        elif os.path.exists("bankruptcy-prevention.csv"):
            file_source = "bankruptcy-prevention.csv"
            
    # Detect file name or type
    filename = ""
    if hasattr(file_source, "name"):
        filename = file_source.name.lower()
    elif isinstance(file_source, str):
        filename = file_source.lower()

    data = None

    # Strategy 1: CSV / TSV / Text by extension
    if filename.endswith(".csv") or filename.endswith(".tsv") or filename.endswith(".txt"):
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        try:
            data = pd.read_csv(file_source)
        except Exception:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            data = pd.read_csv(file_source, sep=";")
    # Strategy 2: Excel by extension
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        try:
            data = pd.read_excel(file_source, sheet_name='bankruptcy-prevention', engine='openpyxl')
        except Exception:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            data = pd.read_excel(file_source, engine='openpyxl')
    # Strategy 3: Multi-attempt fallback for arbitrary buffers
    else:
        try:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            data = pd.read_csv(file_source)
        except Exception:
            try:
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                data = pd.read_excel(file_source, engine='openpyxl')
            except Exception:
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                data = pd.read_csv(file_source, sep=";")

    if data is None or len(data) == 0:
        raise ValueError("The provided file could not be parsed into a valid DataFrame.")

    # Handle semicolon separated dataset format if present
    if len(data.columns) == 1 or ';' in str(data.iloc[0, 0]):
        if data.iloc[:, 0].dtype != object:
            data.iloc[:, 0] = data.iloc[:, 0].astype(str)
        data_split = data.iloc[:, 0].str.split(';', expand=True)
        if data_split.shape[1] >= 7:
            data_split.columns = [
                "industrial_risk", "management_risk", "financial_flexibility",
                "credibility", "competitiveness", "operating_risk", "class"
            ] + list(data_split.columns[7:])
        data = data_split

    # Clean and normalize column names
    data.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in data.columns]

    # Required risk columns
    req_cols = ["industrial_risk", "management_risk", "financial_flexibility",
                "credibility", "competitiveness", "operating_risk"]
    
    # Fill any missing required numeric columns
    for col in req_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0.5)
        else:
            data[col] = 0.5

    # Process class column if present
    if "class" in data.columns:
        data['class'] = data['class'].astype(str).str.lower().str.strip().map({
            'non-bankruptcy': 0, 'bankruptcy': 1, '0': 0, '1': 1, '0.0': 0, '1.0': 1,
            'healthy': 0, 'distress': 1, 'non_bankruptcy': 0, 'bankrupt': 1, 'non-bankrupt': 0,
            'stable': 0, 'positive': 0, 'negative': 1
        }).fillna(0).astype(int)
    else:
        # Generate target class based on financial flexibility & credibility heuristics if missing
        data['class'] = np.where((data['financial_flexibility'] < 0.4) & (data['credibility'] < 0.4), 1, 0)
    
    return data[req_cols + ['class']]

# ---------------------- Train Multiple Models ----------------------
@st.cache_resource
def train_models(data):
    req_cols = ['industrial_risk', 'management_risk', 'financial_flexibility',
                'credibility', 'competitiveness', 'operating_risk']
    X = data[req_cols]
    y = data['class'].astype(int)

    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min()) if len(class_counts) >= 2 else 0
    can_stratify = (len(class_counts) >= 2) and (min_class_count >= 2) and (len(y) >= 10)

    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
        )
    elif len(X) >= 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42, stratify=None
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    models = {
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', max_depth=4, n_estimators=100),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=200)
    }

    trained_models = {}
    model_scores = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
        except Exception:
            model.fit(X, [0, 1] + [0]*(len(X)-2) if len(X) >= 2 else [0])
            
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") and len(getattr(model, 'classes_', [])) > 1 else y_pred

        trained_models[name] = model
        
        cv_folds = min(5, min_class_count, len(y_train)) if min_class_count >= 2 else 0
        if cv_folds >= 2:
            try:
                cv = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            except Exception:
                cv = np.array([float(accuracy_score(y_test, y_pred))])
        else:
            cv = np.array([float(accuracy_score(y_test, y_pred))])

        if len(np.unique(y_test)) > 1 and hasattr(model, 'predict_proba') and len(getattr(model, 'classes_', [])) > 1:
            try:
                auc_val = float(roc_auc_score(y_test, y_prob))
            except Exception:
                auc_val = float(accuracy_score(y_test, y_pred))
        else:
            auc_val = float(accuracy_score(y_test, y_pred))

        model_scores[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': auc_val,
            'cv_scores': cv,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    return trained_models, model_scores, X_test, y_test

# ---------------------- Prediction Functions ----------------------
def predict_bankruptcy(model, features):
    df = pd.DataFrame([features], columns=[
        'industrial_risk', 'management_risk', 'financial_flexibility',
        'credibility', 'competitiveness', 'operating_risk'
    ])
    prediction = int(model.predict(df)[0])
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df)[0]
    else:
        probability = np.array([1.0 - prediction, float(prediction)])
    return prediction, probability

def get_credit_tier(prob_bankruptcy):
    """Determine credit rating tier based on bankruptcy probability"""
    if prob_bankruptcy < 0.05:
        return "AAA", "Prime Financial Health", "#10b981"
    elif prob_bankruptcy < 0.15:
        return "AA", "High Grade Quality", "#059669"
    elif prob_bankruptcy < 0.30:
        return "A", "Upper Medium Grade", "#3b82f6"
    elif prob_bankruptcy < 0.45:
        return "BBB", "Investment Grade Solvency", "#f59e0b"
    elif prob_bankruptcy < 0.60:
        return "BB", "Speculative / Vulnerable", "#f97316"
    elif prob_bankruptcy < 0.75:
        return "B", "Highly Speculative", "#ea580c"
    elif prob_bankruptcy < 0.90:
        return "CCC", "Substantial Risk of Default", "#dc2626"
    else:
        return "D", "In Default / Near Failure", "#991b1b"

def risk_assessment(features):
    risks = []
    feature_names = ['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                    'Credibility', 'Competitiveness', 'Operating Risk']
    
    for i, (name, value) in enumerate(zip(feature_names, features)):
        if i == 2:  # Financial flexibility (higher = better)
            if value <= 0.2:
                risks.append(("🚨 Critical Deficit", f"Critically Low {name} ({value:.1f}) - Inability to handle liquidity shocks", "high"))
            elif value <= 0.5:
                risks.append(("⚠️ Moderate Concern", f"Restricted {name} ({value:.1f}) - Buffer against market downturns is slim", "medium"))
        else:
            if value >= 0.8:
                risks.append(("🚨 High Exposure", f"Elevated {name} ({value:.1f}) - Sector/Operational vulnerability", "high"))
            elif value >= 0.5:
                risks.append(("⚠️ Moderate Exposure", f"Notable {name} ({value:.1f}) - Requires active risk hedging", "medium"))
    
    return risks

# ---------------------- Visualizations ----------------------
def create_radar_chart(features):
    categories = ['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                 'Credibility', 'Competitiveness', 'Operating Risk']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=features,
        theta=categories,
        fill='toself',
        name='Evaluated Company',
        line=dict(color='#6366f1', width=2),
        fillcolor='rgba(99, 102, 241, 0.25)'
    ))
    
    # Industry Average
    avg_features = [0.4, 0.35, 0.65, 0.70, 0.60, 0.40]
    fig.add_trace(go.Scatterpolar(
        r=avg_features,
        theta=categories,
        fill='toself',
        name='Industry Median',
        line=dict(color='#f59e0b', width=1.5, dash='dash'),
        fillcolor='rgba(245, 158, 11, 0.15)'
    ))

    # Top Quartile Benchmark
    top_features = [0.15, 0.10, 0.90, 0.95, 0.90, 0.15]
    fig.add_trace(go.Scatterpolar(
        r=top_features,
        theta=categories,
        fill='none',
        name='Top Quartile Benchmark',
        line=dict(color='#10b981', width=1.5, dash='dot')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=dict(text="Multidimensional Risk Radar Profile", font=dict(size=15, weight="bold")),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def sensitivity_analysis(model, features, feat1_idx, feat2_idx):
    x_range = np.linspace(0, 1, 11)
    y_range = np.linspace(0, 1, 11)
    z = np.zeros((11, 11))
    
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            temp_features = list(features)
            temp_features[feat1_idx] = x
            temp_features[feat2_idx] = y
            _, prob = predict_bankruptcy(model, temp_features)
            z[j, i] = prob[1]
            
    return x_range, y_range, z

# ---------------------- Main Application ----------------------
def main():
    # Top Navigation & Header
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.markdown("""
            <div class="main-header">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 2.8rem;">🛡️</span>
                    <div>
                        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">FinShield AI Enterprise</h1>
                        <p style="margin: 0; opacity: 0.85; font-size: 1rem;">Machine Learning-Driven Corporate Bankruptcy & Distress Prediction System</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with top_c2:
        st.markdown("""
            <div class="stat-card" style="text-align: center; height: 85%;">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase;">System Status</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #059669; margin-top: 4px;">🟢 Operational</div>
                <div class="user-badge" style="margin: 8px auto 0 auto;">4 Models Active</div>
            </div>
        """, unsafe_allow_html=True)

    # Sidebar: Controls & Financial Risk Inputs
    st.sidebar.markdown("### 🎛️ AI Control Panel")
    
    model_choice = st.sidebar.selectbox(
        "🧠 Active Prediction Model",
        ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"],
        help="Select the underlying machine learning classification algorithm."
    )
    
    interactive_mode = st.sidebar.toggle("⚡ Real-Time Auto-Compute", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Financial Risk Parameters")
    
    industrial_risk = st.sidebar.slider("🏭 Industrial Risk", 0.0, 1.0, 0.5, step=0.1, help="Sector volatility and market cyclicality")
    management_risk = st.sidebar.slider("👥 Management Risk", 0.0, 1.0, 0.5, step=0.1, help="Corporate governance and executive decision efficacy")
    financial_flexibility = st.sidebar.slider("💰 Financial Flexibility", 0.0, 1.0, 0.5, step=0.1, help="Ability to raise capital & adapt to shocks (Higher = Healthier)")
    credibility = st.sidebar.slider("🏆 Credibility Score", 0.0, 1.0, 0.5, step=0.1, help="Creditworthiness, market trust, and payment history")
    competitiveness = st.sidebar.slider("⚡ Competitiveness", 0.0, 1.0, 0.5, step=0.1, help="Market share and pricing power")
    operating_risk = st.sidebar.slider("⚙️ Operating Risk", 0.0, 1.0, 0.5, step=0.1, help="Supply chain, internal processes, and cost structure")
    
    st.sidebar.markdown("### 🎯 Quick Stress Archetypes")
    preset_cols = st.sidebar.columns(3)
    if preset_cols[0].button("🟢 Prime"):
        industrial_risk, management_risk, operating_risk = 0.1, 0.1, 0.1
        financial_flexibility, credibility, competitiveness = 0.9, 0.9, 0.9
    if preset_cols[1].button("🟡 Average"):
        industrial_risk, management_risk, operating_risk = 0.5, 0.5, 0.5
        financial_flexibility, credibility, competitiveness = 0.5, 0.5, 0.5
    if preset_cols[2].button("🔴 Distress"):
        industrial_risk, management_risk, operating_risk = 0.9, 0.8, 0.9
        financial_flexibility, credibility, competitiveness = 0.1, 0.2, 0.1

    # Data Loader
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Dataset Source")
    uploaded_file = st.sidebar.file_uploader("Upload custom Excel / CSV", type=["xlsx", "csv", "tsv", "txt"])
    
    try:
        data = load_data(uploaded_file)
        if uploaded_file is not None:
            st.sidebar.success(f"✅ Loaded: {uploaded_file.name} ({len(data)} rows)")
        else:
            st.sidebar.info(f"📊 Default Dataset ({len(data)} rows)")
    except Exception as e:
        st.sidebar.error(f"❌ Upload error: {e}. Falling back to default.")
        data = load_data(None)

    trained_models, model_scores, X_test, y_test = train_models(data)

    # Current Evaluation Feature Vector
    features = [industrial_risk, management_risk, financial_flexibility,
                credibility, competitiveness, operating_risk]
    
    selected_model = trained_models[model_choice]
    prediction, probability = predict_bankruptcy(selected_model, features)
    distress_prob = float(probability[1])
    stability_score = float(1.0 - distress_prob)
    credit_tier, tier_desc, tier_color = get_credit_tier(distress_prob)

    # Log to history with native Python serializable types
    log_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "model": str(model_choice),
        "distress_prob": round(float(distress_prob), 4),
        "prediction": "Bankruptcy Risk" if int(prediction) == 1 else "Stable",
        "tier": str(credit_tier)
    }
    if not st.session_state.history or st.session_state.history[-1]["distress_prob"] != log_entry["distress_prob"] or st.session_state.history[-1]["model"] != log_entry["model"]:
        st.session_state.history.append(log_entry)

    # Tabs Interface
    tab_pred, tab_adv, tab_stress, tab_batch, tab_models, tab_advisory, tab_data = st.tabs([
        "🔮 Real-Time Predictor",
        "📊 Advanced Analytics",
        "🎯 Stress Testing & Monte Carlo",
        "📂 Batch Processing",
        "📈 ML Benchmark",
        "💡 AI Advisory",
        "📋 Data & Audit Trail"
    ])

    # ---------------- TAB 1: REAL-TIME PREDICTOR ----------------
    with tab_pred:
        col_res1, col_res2 = st.columns([1.8, 1.2])

        with col_res1:
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-card danger-gradient">
                        <div style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">🚨 DISTRESS RISK DETECTED</div>
                        <div style="font-size: 1.15rem; margin-top: 8px; opacity: 0.95;">Estimated Probability of Bankruptcy: <strong>{distress_prob:.1%}</strong></div>
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 4px;">Classified by {model_choice} AI Model</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-card success-gradient">
                        <div style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px;">✅ FINANCIALLY STABLE</div>
                        <div style="font-size: 1.15rem; margin-top: 8px; opacity: 0.95;">Financial Solvency Score: <strong>{stability_score:.1%}</strong></div>
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 4px;">Classified by {model_choice} AI Model</div>
                    </div>
                """, unsafe_allow_html=True)

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=distress_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>Bankruptcy Probability Gauge (%)</b>", 'font': {'size': 18}},
                delta={'reference': 50, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
                    'bar': {'color': "#dc2626" if prediction == 1 else "#059669", 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.25)"},
                        {'range': [20, 45], 'color': "rgba(59, 130, 246, 0.25)"},
                        {'range': [45, 70], 'color': "rgba(245, 158, 11, 0.25)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "#0f172a", 'width': 4},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk factors
            identified_risks = risk_assessment(features)
            st.markdown("#### 🔍 Primary Risk Factor Diagnostic")
            if identified_risks:
                for badge, desc, level in identified_risks:
                    pill_class = "risk-pill-high" if level == "high" else "risk-pill-medium"
                    st.markdown(f"""
                        <div class="stat-card" style="margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                            <span style="font-weight: 600; color: #1e293b;">{desc}</span>
                            <span class="risk-pill {pill_class}">{badge}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🌟 All fundamental risk indicators are within safe thresholds.")

        with col_res2:
            # Credit Tier Card
            st.markdown(f"""
                <div class="stat-card" style="text-align: center; border-top: 5px solid {tier_color};">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Credit Health Rating</div>
                    <div class="tier-badge" style="color: {tier_color};">{credit_tier}</div>
                    <div style="font-size: 0.95rem; font-weight: 600; color: #334155;">{tier_desc}</div>
                </div>
            """, unsafe_allow_html=True)

            # Radar Chart
            radar_fig = create_radar_chart(features)
            st.plotly_chart(radar_fig, use_container_width=True, height=350)

            # Interactive Celebration
            if prediction == 0 and distress_prob < 0.15:
                if st.button("🎉 Validate & Celebrate Solvency!", use_container_width=True):
                    st.balloons()
                    st.toast("Corporate health verified: Tier-1 Solvency!", icon="✨")

    # ---------------- TAB 2: ADVANCED ANALYTICS ----------------
    with tab_adv:
        st.markdown("### 📊 Interactive Sensitivity & Interaction Heatmap")
        st.write("Analyze how changing two risk parameters in parallel alters the overall probability of bankruptcy.")
        
        feature_names = ['Industrial Risk', 'Management Risk', 'Financial Flexibility',
                         'Credibility', 'Competitiveness', 'Operating Risk']
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            feat1 = st.selectbox("X-Axis Parameter:", feature_names, index=2)
        with col_s2:
            feat2 = st.selectbox("Y-Axis Parameter:", feature_names, index=1)
            
        f1_idx = feature_names.index(feat1)
        f2_idx = feature_names.index(feat2)
        
        x_rng, y_rng, z_data = sensitivity_analysis(selected_model, features, f1_idx, f2_idx)
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f"{x:.1f}" for x in x_rng],
            y=[f"{y:.1f}" for y in y_rng],
            colorscale='RdYlGn_r',
            hovertemplate=f"<b>{feat1}</b>: %{{x}}<br><b>{feat2}</b>: %{{y}}<br><b>Risk Probability</b>: %{{z:.1%}}<extra></extra>"
        ))
        fig_heat.update_layout(
            title=f"<b>Risk Interaction Landscape: {feat1} vs {feat2}</b>",
            xaxis_title=feat1,
            yaxis_title=feat2,
            height=450
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("#### 📈 Feature Importance Decomposition")
            if hasattr(selected_model, "feature_importances_"):
                imp = selected_model.feature_importances_
            else:
                imp = np.abs(selected_model.coef_[0]) if hasattr(selected_model, "coef_") else [1/6]*6
            
            fig_imp = px.bar(
                x=imp, y=feature_names, orientation="h",
                color=imp, color_continuous_scale="Purples",
                labels={"x": "Relative Weight / Importance", "y": "Feature"},
                title=f"<b>Feature Importance ({model_choice})</b>"
            )
            fig_imp.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_a2:
            st.markdown("#### 📉 Projected Trajectory Simulation (30 Days)")
            dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
            np.random.seed(42)
            sim_probs = np.clip([distress_prob + np.random.normal(0, 0.02) for _ in range(30)], 0, 1)
            
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=dates, y=sim_probs, mode='lines+markers',
                name='Simulated Probability',
                line=dict(color="#ef4444" if distress_prob > 0.5 else "#10b981", width=2.5)
            ))
            fig_traj.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b", annotation_text="Default Threshold (50%)")
            fig_traj.update_layout(title="<b>Projected 30-Day Risk Evolution</b>", yaxis_title="Probability", height=350)
            st.plotly_chart(fig_traj, use_container_width=True)

    # ---------------- TAB 3: STRESS TESTING & MONTE CARLO ----------------
    with tab_stress:
        st.markdown("### 🎲 Monte Carlo Risk Simulation")
        st.write("Quantify probability confidence bounds by simulating thousands of randomized macroeconomic & operational fluctuations.")
        
        col_m1, col_m2 = st.columns([1, 3])
        with col_m1:
            num_sims = st.slider("Simulation Iterations", 100, 2000, 1000, step=100)
            volatility = st.slider("Market Volatility (σ)", 0.05, 0.30, 0.12, step=0.01)
            run_sim = st.button("🚀 Execute Monte Carlo", use_container_width=True)
            
        with col_m2:
            if run_sim or "mc_results" not in st.session_state:
                np.random.seed(42)
                mc_res = []
                for _ in range(num_sims):
                    perturbed = [max(0.0, min(1.0, f + np.random.normal(0, volatility))) for f in features]
                    _, p = predict_bankruptcy(selected_model, perturbed)
                    mc_res.append(p[1])
                st.session_state.mc_results = mc_res

            results = st.session_state.mc_results
            mean_risk = np.mean(results)
            var_95 = np.percentile(results, 95)
            
            fig_mc = px.histogram(
                x=results, nbins=40,
                title=f"<b>Monte Carlo Risk Distribution ({len(results):,} Runs)</b>",
                labels={"x": "Simulated Bankruptcy Probability", "y": "Frequency"},
                color_discrete_sequence=["#6366f1"]
            )
            fig_mc.add_vline(x=mean_risk, line_dash="solid", line_color="#0f172a", annotation_text=f"Mean: {mean_risk:.1%}")
            fig_mc.add_vline(x=var_95, line_dash="dash", line_color="#dc2626", annotation_text=f"95% VaR: {var_95:.1%}")
            fig_mc.update_layout(height=380)
            st.plotly_chart(fig_mc, use_container_width=True)

        st.markdown("---")
        st.markdown("#### ⚡ Macroeconomic Shock Stress Test")
        scenarios = {
            "Baseline": features,
            "Mild Industry Downturn (+20% Ind Risk)": [min(1.0, features[0]+0.2), features[1], features[2], features[3], features[4], features[5]],
            "Liquidity Squeeze (-30% Fin Flexibility)": [features[0], features[1], max(0.0, features[2]-0.3), features[3], features[4], features[5]],
            "Severe Multi-Shock Crisis": [min(1.0, features[0]+0.3), min(1.0, features[1]+0.2), max(0.0, features[2]-0.4), max(0.0, features[3]-0.3), max(0.0, features[4]-0.3), min(1.0, features[5]+0.3)]
        }
        
        sc_cols = st.columns(4)
        for i, (sc_name, sc_feat) in enumerate(scenarios.items()):
            _, sc_p = predict_bankruptcy(selected_model, sc_feat)
            sc_color = "#dc2626" if sc_p[1] > 0.5 else "#059669"
            with sc_cols[i]:
                st.markdown(f"""
                    <div class="stat-card" style="border-left: 4px solid {sc_color};">
                        <div style="font-size: 0.85rem; font-weight: 700; color: #64748b;">{sc_name}</div>
                        <div style="font-size: 1.5rem; font-weight: 800; color: {sc_color}; margin-top: 6px;">{sc_p[1]:.1%}</div>
                        <div style="font-size: 0.8rem; color: #475569;">{'🚨 Distress' if sc_p[1] > 0.5 else '✅ Solvency'}</div>
                    </div>
                """, unsafe_allow_html=True)

    # ---------------- TAB 4: BATCH PROCESSING ----------------
    with tab_batch:
        st.markdown("### 📂 Bulk Company Risk Scoring")
        st.write("Upload an Excel or CSV file containing a list of companies to score them in bulk.")
        
        batch_file = st.file_uploader("Upload Company Portfolio File (CSV / XLSX)", type=["csv", "xlsx"], key="batch_uploader")
        
        if batch_file is not None:
            try:
                b_name = batch_file.name.lower() if hasattr(batch_file, "name") else ""
                if b_name.endswith(".csv") or b_name.endswith(".tsv") or b_name.endswith(".txt"):
                    batch_df = pd.read_csv(batch_file)
                else:
                    try:
                        batch_df = pd.read_excel(batch_file, engine='openpyxl')
                    except Exception:
                        if hasattr(batch_file, "seek"):
                            batch_file.seek(0)
                        batch_df = pd.read_csv(batch_file)
                
                # Normalize column headers
                batch_df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in batch_df.columns]
                st.write(f"Loaded {len(batch_df)} companies.")
                
                req_cols = ['industrial_risk', 'management_risk', 'financial_flexibility', 'credibility', 'competitiveness', 'operating_risk']
                if all(col in batch_df.columns for col in req_cols):
                    preds = []
                    probs = []
                    tiers = []
                    for _, row in batch_df[req_cols].iterrows():
                        p, pr = predict_bankruptcy(selected_model, row.values)
                        preds.append("Distress" if p == 1 else "Stable")
                        probs.append(round(pr[1] * 100, 2))
                        tier, _, _ = get_credit_tier(pr[1])
                        tiers.append(tier)
                        
                    batch_df["Bankruptcy_Risk_Pct"] = probs
                    batch_df["Prediction"] = preds
                    batch_df["Credit_Rating"] = tiers
                    
                    st.dataframe(batch_df, use_container_width=True)
                    
                    csv_export = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored Portfolio (CSV)",
                        data=csv_export,
                        file_name=f"finshield_portfolio_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Missing required columns. Please ensure the file contains: {', '.join(req_cols)}")
            except Exception as e:
                st.error(f"Error parsing batch file: {e}")
        else:
            st.info("💡 Tip: You can download sample portfolio template below to test batch scoring.")
            sample_template = pd.DataFrame({
                "company_name": ["Alpha Corp", "Beta LLC", "Gamma Inc", "Delta Ltd"],
                "industrial_risk": [0.2, 0.8, 0.5, 0.9],
                "management_risk": [0.1, 0.7, 0.5, 0.8],
                "financial_flexibility": [0.9, 0.2, 0.6, 0.1],
                "credibility": [0.8, 0.3, 0.6, 0.2],
                "competitiveness": [0.9, 0.2, 0.7, 0.1],
                "operating_risk": [0.2, 0.8, 0.4, 0.9]
            })
            st.dataframe(sample_template, use_container_width=True)
            st.download_button(
                "📥 Download Sample Template (CSV)",
                sample_template.to_csv(index=False).encode('utf-8'),
                "finshield_sample_template.csv",
                "text/csv"
            )

    # ---------------- TAB 5: ML BENCHMARK ----------------
    with tab_models:
        st.markdown("### 📈 Comprehensive Machine Learning Benchmark")
        
        bench_cols = st.columns(4)
        for i, (m_name, scores) in enumerate(model_scores.items()):
            with bench_cols[i]:
                st.markdown(f"""
                    <div class="stat-card" style="text-align: center;">
                        <div style="font-size: 0.9rem; font-weight: 800; color: #312e81;">{m_name}</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: #0f172a; margin: 4px 0;">{scores['accuracy']:.1%}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">Accuracy Score</div>
                        <hr style="margin: 8px 0; border: none; border-top: 1px solid #e2e8f0;">
                        <div style="font-size: 0.8rem; color: #475569;">ROC-AUC: <strong>{scores['roc_auc']:.3f}</strong></div>
                        <div style="font-size: 0.8rem; color: #475569;">F1-Score: <strong>{scores['f1']:.3f}</strong></div>
                    </div>
                """, unsafe_allow_html=True)
                
        st.markdown("---")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            models_list = list(model_scores.keys())
            accs = [model_scores[m]['accuracy'] for m in models_list]
            aucs = [model_scores[m]['roc_auc'] for m in models_list]
            f1s = [model_scores[m]['f1'] for m in models_list]
            
            fig_compare = go.Figure(data=[
                go.Bar(name='Accuracy', x=models_list, y=accs, marker_color='#6366f1'),
                go.Bar(name='ROC-AUC', x=models_list, y=aucs, marker_color='#10b981'),
                go.Bar(name='F1-Score', x=models_list, y=f1s, marker_color='#f59e0b')
            ])
            fig_compare.update_layout(barmode='group', title="<b>Model Performance Metric Comparison</b>", height=380)
            st.plotly_chart(fig_compare, use_container_width=True)

        with col_b2:
            st.markdown(f"#### 📌 Confusion Matrix: {model_choice}")
            cm = confusion_matrix(y_test, model_scores[model_choice]['y_pred'], labels=[0, 1])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=['Non-Bankruptcy', 'Bankruptcy'],
                y=['Non-Bankruptcy', 'Bankruptcy'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(title=f"<b>Confusion Matrix ({model_choice})</b>", height=380)
            st.plotly_chart(fig_cm, use_container_width=True)

    # ---------------- TAB 6: AI ADVISORY ----------------
    with tab_advisory:
        st.markdown("### 💡 Executive Financial Advisory & Mitigation Engine")
        st.write("Automated strategic recommendations tailored to the specific vulnerabilities identified.")

        adv_col1, adv_col2 = st.columns([1, 1])
        with adv_col1:
            st.markdown("#### 🎯 Vulnerability Breakdown")
            cat_health = (financial_flexibility + credibility) / 2
            cat_market = (competitiveness + (1.0 - industrial_risk)) / 2
            cat_ops = ((1.0 - operating_risk) + (1.0 - management_risk)) / 2

            st.write(f"**Financial Solvency Buffer**: {cat_health:.1%}")
            st.progress(cat_health)
            st.write(f"**Market & Sector Strength**: {cat_market:.1%}")
            st.progress(cat_market)
            st.write(f"**Operational Resilience**: {cat_ops:.1%}")
            st.progress(cat_ops)

        with adv_col2:
            st.markdown("#### 📋 Recommended Strategic Interventions")
            recommendations = []
            if financial_flexibility < 0.4:
                recommendations.append("💰 **Restructure Debt Portfolio**: Extend short-term debt maturities and secure emergency revolving credit facilities.")
            if credibility < 0.5:
                recommendations.append("🏆 **Rebuild Market Transparency**: Implement enhanced quarterly investor disclosures and audited compliance reporting.")
            if industrial_risk > 0.6:
                recommendations.append("🏭 **Hedge Sector Concentration**: Diversify customer segments and utilize derivative hedging against sector commodity swings.")
            if competitiveness < 0.5:
                recommendations.append("⚡ **Refocus Core Value Proposition**: Divest low-margin business units to reinvest in high-margin core competencies.")
            if operating_risk > 0.6:
                recommendations.append("⚙️ **Lean Operational Restructuring**: Audit fixed-cost overheads and implement automation in core supply chains.")

            if not recommendations:
                st.success("🌟 The enterprise demonstrates resilient financial health. Maintain current fiscal governance and liquidity buffers.")
            else:
                for rec in recommendations:
                    st.info(rec)

    # ---------------- TAB 7: DATA & AUDIT TRAIL ----------------
    with tab_data:
        st.markdown("### 📋 Training Dataset & Live User Audit Log")
        
        tab_d1, tab_d2 = st.tabs(["📊 Training Dataset Overview", "📝 User Session Audit Trail"])
        with tab_d1:
            st.dataframe(data.head(20), use_container_width=True)
            st.write(f"Total Instances: {len(data)} | Class Distribution: {data['class'].value_counts().to_dict()}")
            
            corr = data.corr()
            fig_corr = px.imshow(corr, title="<b>Feature Correlation Matrix</b>", color_continuous_scale="RdBu_r", text_auto=".2f")
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab_d2:
            if st.session_state.history:
                hist_df = pd.DataFrame(st.session_state.history)
                st.dataframe(hist_df, use_container_width=True)
                st.download_button(
                    "📥 Export Audit Trail (JSON)",
                    json.dumps(st.session_state.history, indent=2, default=str),
                    f"finshield_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            else:
                st.info("No audit entries recorded yet in this session.")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #64748b; font-size: 0.9rem; padding: 20px;'>
            🛡️ <strong>FinShield AI Enterprise Edition</strong> | Developed by <strong>Abhinay Patel</strong> (Backend & Architect) & <strong>Sujal Gupta</strong> (Frontend & ML)<br>
            <span style='font-size: 0.8rem;'>Protected under MIT License | For Enterprise Risk Management and Research</span>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()