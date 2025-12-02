import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(page_title="ARDS Prediction Tool", layout="wide")

# CSS Styling
st.markdown("""
<style>
    /* 1. Layout Adjustments */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* 【关键修改 1】全局垂直间距压缩到极致 */
    div[data-testid="stVerticalBlock"] {
        gap: 0.05rem !important; 
    }

    /* 标题样式 */
    .main-header {
        text-align: center; 
        color: #000000; 
        margin-bottom: 12px; 
        font-weight: 600;
        font-size: 30px;
    }

    /* 2. Slider Theme - Blue */
    div.stSlider > div[data-baseweb = "slider"] > div > div > div > div {
        background-color: #007bff !important;
    }

    /* 3. Custom Label Style */
    .custom-label {
        font-size: 20px !important;  
        font-weight: 500 !important; 
        color: #333 !important;   
        margin-bottom: 0px !important; 
        line-height: 1.2;
        /* 【关键修改 2】标签上方的间距大大减小，让列表更紧凑 */
        margin-top: 3px !important;   
    }

    /* 4. Value display style */
    .value-display {
        font-size: 22px; 
        font-weight: bold; 
        color: #007bff;
        text-align: left;
        /* 微调对齐 */
        padding-top: 12px; 
    }

    /* 增强边框样式 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 2px solid #cccccc !important; 
        border-radius: 12px !important;      
        padding: 15px !important;             
        background-color: #ffffff;            
    }

    /* Hide standard Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Buttons */
    div.stButton > button {
        border-radius: 10px;
        height: 2em; 
        font-weight: bold;
        font-size: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Variable Settings
# ==========================================
CATEGORICAL_VARS = ["ventilation"]
VAR_SETTINGS = {
    # Name: [Min, Max, Default, Step]
    "BMI": [5.0, 40.0, 25.0, 1.0],
    "PO2": [0.0, 400.0, 100.0, 1.0],
    "SOFA": [0, 20, 8, 1],
    "SAPS II": [0, 120, 63, 1],
    "RR": [0, 40, 18, 1],
    "PCO2": [0.0, 110.0, 40.0, 1.0],
}


# ==========================================
3. Load Model (FIXED FOR CLOUD DEPLOYMENT)
# ==========================================
@st.cache_resource
def load_model():
    # 🚨 移除绝对路径 F:/Python_work/articleagain
    # 假设模型文件在 GitHub 仓库的根目录
    try:
        model = joblib.load("rf_model_deploy.pkl")
        features = joblib.load("feature_names.pkl")
        return model, features
    except Exception as e:
        # 在云端显示加载失败的原因，帮助调试
        st.error(f"Deployment Error: Failed to load model files. Please ensure rf_model_deploy.pkl and feature_names.pkl are in the GitHub repository root. Details: {e}")
        return None, None

model, feature_names = load_model()


# ==========================================
# 4. Helper Functions
# ==========================================
def reset_inputs():
    if feature_names:
        for f in feature_names:
            if f in VAR_SETTINGS:
                st.session_state[f] = float(VAR_SETTINGS[f][2])
            elif f in CATEGORICAL_VARS:
                st.session_state[f] = 0


# ==========================================
# 5. Interface Layout
# ==========================================

st.markdown(
    "<div class='main-header'>Prediction probability of ARDS based on RF model</div>",
    unsafe_allow_html=True)

user_input = {}

# Layout: Left (Controls) - Right (Result)
# 【关键修改 3】gap="small" 让左右两栏靠得更近
col_input, col_result = st.columns([1.8, 1], gap="small")

# --- Left Column: Inputs ---
with col_input:
    st.markdown(
        "<h3 style='color: #444; margin-bottom: 5px; margin-top:0px; font-weight: 600;'>Parameter Settings</h3>",
        unsafe_allow_html=True)

    with st.container(border=True):
        if feature_names:
            numerical_features = [f for f in feature_names if f not in CATEGORICAL_VARS]

            # 1. Numerical Inputs
            for feature in numerical_features:
                settings = VAR_SETTINGS.get(feature, [0.0, 100.0, 0.0, 1.0])
                min_v, max_v, def_v, step_v = settings

                if feature not in st.session_state:
                    st.session_state[feature] = float(def_v)

                c1, c2 = st.columns([3.5, 1])

                with c1:
                    st.markdown(f"<div class='custom-label'>{feature}</div>", unsafe_allow_html=True)
                    val = st.slider("", min_value=float(min_v), max_value=float(max_v), step=float(step_v), key=feature,
                                    label_visibility="collapsed")
                with c2:
                    st.markdown(f"<div class='value-display'>{val}</div>", unsafe_allow_html=True)

                user_input[feature] = val

            # 2. Categorical Inputs
            st.write("")
            cat_cols = st.columns(2)
            for idx, feature in enumerate(CATEGORICAL_VARS):
                if feature in feature_names:
                    if feature not in st.session_state:
                        st.session_state[feature] = 0

                    with cat_cols[idx % 2]:
                        st.markdown(f"<div class='custom-label'>{feature}</div>", unsafe_allow_html=True)
                        user_input[feature] = st.radio("", options=[0, 1],
                                                       format_func=lambda x: "No" if x == 0 else "Yes",
                                                       horizontal=True, key=feature, label_visibility="collapsed")

# --- Right Column: Results ---
with col_result:
    # 顶部留白
    st.markdown("<div style='height: 45px'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<h3 style='text-align: center; color: #444; margin-bottom: 0px;'>Result</h3>",
                    unsafe_allow_html=True)

        chart_placeholder = st.empty()
        result_spacer = st.empty()
        result_placeholder = st.empty()

        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            if model:
                try:
                    input_df = pd.DataFrame([user_input], columns=feature_names)
                    try:
                        prediction_val = model.predict_proba(input_df)[0][1]
                    except:
                        prediction_val = model.predict(input_df)[0]
                    risk_percent = prediction_val * 100
                except Exception as e:
                    st.error(f"Error: {e}")
                    risk_percent = 0

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percent,
                    number={'suffix': "%", 'font': {'size': 45, 'color': "#007bff"}},
                    gauge={
                        'axis': {
                            'range': [0, 100],  # 修正回 0-100
                            'tickwidth': 1,
                            'dtick': 50
                        },
                        'bar': {'color': "#007bff"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#ddd",
                        'steps': [{'range': [0, 100], 'color': '#f8f9fa'}],
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=40, r=50, t=10, b=10))
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                result_spacer.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)

                if risk_percent < 30:
                    result_placeholder.success(f"**Low Probability**: {risk_percent:.1f}%")
                elif risk_percent < 70:
                    result_placeholder.warning(f"**Medium Probability**: {risk_percent:.1f}%")
                else:
                    result_placeholder.error(f"**High Probability**: {risk_percent:.1f}%")
            else:
                st.error("Model not loaded.")
        else:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=0,
                title={'text': "Ready", 'font': {'size': 24, 'color': "#aaa"}},
                gauge={
                    'axis': {'range': [0, 100], 'dtick': 50},
                    'bar': {'color': "#eee"}
                }
            ))
            fig.update_layout(height=230, margin=dict(l=40, r=50, t=10, b=10))
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            result_spacer.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)


        st.button("🔄 Reset Parameters", on_click=reset_inputs, use_container_width=True)
