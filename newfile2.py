from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import os
import pickle

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="NIDS – Network Intrusion Detection",
    layout="wide"
)

# ==============================
# DARK UI CSS
# ==============================
st.markdown("""
<style>
body { background-color: #0b1220; }
.block-container { padding-top: 2rem; }
h1,h2,h3,h4,h5,h6,p,label { color: #e5e7eb !important; }
.card {
    background: linear-gradient(145deg, #0f172a, #020617);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #1e293b;
}
.stButton>button {
    background: linear-gradient(90deg, #22d3ee, #38bdf8);
    color: black;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-weight: 600;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD ARTIFACTS (UNCHANGED)
# ==============================
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(__file__)

    model = load_model(os.path.join(BASE_DIR, "nids_model.h5"))

    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(BASE_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)

    with open(os.path.join(BASE_DIR, "features.pkl"), "rb") as f:
        features = pickle.load(f)

    return model, scaler, encoders, features


model, scaler, encoders, features = load_artifacts()

# ==============================
# NAVIGATION (UPDATED)
# ==============================
if "menu" not in st.session_state:
    st.session_state.menu = "Home"

menu = st.radio(
    "", ["Home", "Analyze", "Batch Analysis"],
    index=["Home", "Analyze", "Batch Analysis"].index(st.session_state.menu),
    horizontal=True
)

st.session_state.menu = menu

# ==============================
# HOME
# ==============================
if menu == "Home":
    st.markdown(
        "<h1 style='text-align:center;'>Network Intrusion "
        "<span style='color:#38bdf8'>Detection System</span></h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;'>Deep Learning model trained on KDD dataset</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 Analyze Single Input"):
            st.session_state.menu = "Analyze"

    with col2:
        if st.button("📊 Batch Analysis"):
            st.session_state.menu = "Batch Analysis"

# ==============================
# ANALYZE (UNCHANGED)
# ==============================
elif menu == "Analyze":
    st.markdown("<h2 style='text-align:center;'>Analyze Network Traffic</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        protocol = st.selectbox("Protocol Type", encoders["protocol_type"].classes_)
        service = st.selectbox("Service", encoders["service"].classes_)
        flag = st.selectbox("Flag", encoders["flag"].classes_)

    with col2:
        src_bytes = st.number_input("Source Bytes", min_value=0, value=232)
        dst_bytes = st.number_input("Destination Bytes", min_value=0, value=8153)
        count = st.number_input("Connection Count", min_value=0, value=10)
        srv_count = st.number_input("Service Count", min_value=0, value=10)

    if st.button("🚀 Analyze"):
        row = dict.fromkeys(features, 0)

        row["protocol_type"] = encoders["protocol_type"].transform([protocol])[0]
        row["service"] = encoders["service"].transform([service])[0]
        row["flag"] = encoders["flag"].transform([flag])[0]

        row["src_bytes"] = src_bytes
        row["dst_bytes"] = dst_bytes
        row["count"] = count
        row["srv_count"] = srv_count

        row.update({
            "serror_rate": 0.0,
            "srv_serror_rate": 0.0,
            "rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "dst_host_count": 255,
            "dst_host_srv_count": 255
        })

        df = pd.DataFrame([row])[features]
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        df_scaled = scaler.transform(df)

        pred = model.predict(df_scaled)[0][0]
        threshold = 0.2

        label = "ATTACK 🚨" if pred > threshold else "NORMAL ✅"

        st.metric("Attack Probability", f"{pred:.4f}")
        st.success(f"Prediction: **{label}**")

# ==============================
# BATCH ANALYSIS (ANN ADDED)
# ==============================
elif menu == "Batch Analysis":
    st.markdown(
        "<h2 style='text-align:center;'>Batch CSV Analysis & NIDS Prediction</h2>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        st.stop()

    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

    st.dataframe(df.head())

    # Classification
    st.subheader("🤖 ANN Classification")

    all_columns = df.columns.tolist()
    target_col = st.selectbox("Select target column", all_columns)

    feature_cols = st.multiselect(
        "Select feature columns",
        [c for c in all_columns if c != target_col]
    )

    model_df = df[[target_col] + feature_cols].dropna().copy()

    # Encoding
    for col in model_df.columns:
        if model_df[col].dtype == object:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))

    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_ann = StandardScaler()
    X_train = scaler_ann.fit_transform(X_train)
    X_test = scaler_ann.transform(X_test)

    # ANN MODEL
    ann_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    ann_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    ann_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred = (ann_model.predict(X_test) > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)

    st.write("Accuracy:", acc)
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()))

