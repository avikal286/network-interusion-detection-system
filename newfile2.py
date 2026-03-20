from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
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
# SESSION STATE FOR NAVIGATION
# ==============================
if "menu" not in st.session_state:
    st.session_state.menu = "Home"

# ==============================
# LOAD ARTIFACTS
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
# NAVIGATION
# ==============================
menu = st.radio(
    "",
    ["Home", "Analyze", "Batch Analysis"],
    horizontal=True,
    index=["Home", "Analyze", "Batch Analysis"].index(st.session_state.menu),
    key="nav_radio"
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

    # ── TWO NAVIGATION BUTTONS ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_mid, col_right = st.columns([1, 2, 1])

    with col_mid:
        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("🔍 Go to Analyze", use_container_width=True):
                st.session_state.menu = "Analyze"
                st.rerun()

        with btn_col2:
            if st.button("📂 Go to Batch Analysis", use_container_width=True):
                st.session_state.menu = "Batch Analysis"
                st.rerun()
    # ────────────────────────────────────────────────────────────────────────

# ==============================
# ANALYZE (SINGLE INPUT)
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
# BATCH ANALYSIS
# ==============================
elif menu == "Batch Analysis":
    st.markdown(
        "<h2 style='text-align:center;'>Batch CSV Analysis & NIDS Prediction</h2>",
        unsafe_allow_html=True
    )

    # --- File upload ---
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("👆 Upload a CSV file to get started.")
        st.stop()

    # --- Load data ---
    try:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("🔎 Data Overview")
    st.dataframe(df.head())

    st.subheader("📈 Data Statistics")
    st.dataframe(df.describe(include="all").T)

    # -------------------------
    # FILTERING
    # -------------------------
    st.subheader("🧪 Filter Data")
    columns = df.columns.tolist()
    filter_col = st.selectbox("Select column to filter", columns)
    filter_val = st.selectbox(
        f"Select value from {filter_col}",
        df[filter_col].dropna().unique()
    )

    filtered_df = df[df[filter_col] == filter_val].copy()
    st.dataframe(filtered_df)

    # -------------------------
    # PLOTTING
    # -------------------------
    st.subheader("📊 Plot Data Overview")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Select X-axis column", numeric_cols)
        y_col = st.selectbox("Select Y-axis column", numeric_cols)
        plot_type = st.radio("Plot type", ["Line", "Pie"], horizontal=True)

        if st.button("Plot Data"):
            plot_df = df[[x_col, y_col]].dropna()

            if plot_type == "Line":
                st.line_chart(plot_df.set_index(x_col))
            else:
                agg = plot_df.groupby(x_col)[y_col].sum()
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(agg, labels=agg.index, autopct="%1.1f%%")
                ax.set_title(f"{y_col} distribution over {x_col}")
                st.pyplot(fig)
    else:
        st.warning("Need at least two numeric columns for plotting.")

    # -------------------------
    # CLASSIFICATION  ← ANN
    # -------------------------
    st.subheader("🤖 Classification: Train & Evaluate (ANN)")

    all_columns = df.columns.tolist()
    target_col = st.selectbox("Select target column", all_columns)
    feature_cols = st.multiselect(
        "Select feature columns",
        [c for c in all_columns if c != target_col],
        default=[c for c in numeric_cols if c != target_col][:2]
    )

    if not feature_cols:
        st.warning("Please select at least one feature column.")
        st.stop()

    model_df = df[[target_col] + feature_cols].dropna().copy()

    # Encode categorical columns safely
    clf_encoders = {}
    for col in [target_col] + feature_cols:
        if model_df[col].dtype == object or not pd.api.types.is_numeric_dtype(model_df[col]):
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            clf_encoders[col] = le

    X = model_df[feature_cols].values
    y = model_df[target_col].values

    test_size = st.slider("Test set proportion", 0.1, 0.5, 0.2)
    stratify_arr = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_arr
    )

    # ── Scale features ──────────────────────────────────────────────────────
    ann_scaler = StandardScaler()
    X_train_sc = ann_scaler.fit_transform(X_train)
    X_test_sc  = ann_scaler.transform(X_test)

    # ── Determine task type ─────────────────────────────────────────────────
    n_classes = len(np.unique(y))
    is_binary = n_classes == 2

    if is_binary:
        y_train_enc = y_train.astype(float)
        y_test_enc  = y_test.astype(float)
        output_units      = 1
        output_activation = "sigmoid"
        loss_fn           = "binary_crossentropy"
    else:
        y_train_enc = to_categorical(y_train, num_classes=n_classes)
        y_test_enc  = to_categorical(y_test,  num_classes=n_classes)
        output_units      = n_classes
        output_activation = "softmax"
        loss_fn           = "categorical_crossentropy"

    # ── Build ANN ───────────────────────────────────────────────────────────
    ann = Sequential([
        Dense(128, activation="relu", input_shape=(X_train_sc.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(output_units, activation=output_activation)
    ])

    ann.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    with st.spinner("🧠 Training ANN…"):
        history = ann.fit(
            X_train_sc, y_train_enc,
            validation_split=0.1,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

    # ── Predict ─────────────────────────────────────────────────────────────
    raw_preds = ann.predict(X_test_sc)

    if is_binary:
        y_pred = (raw_preds.flatten() > 0.5).astype(int)
    else:
        y_pred = np.argmax(raw_preds, axis=1)

    # ── Metrics ─────────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True)
    ).transpose()

    cm = confusion_matrix(y_test, y_pred)

    # ── Training curve ──────────────────────────────────────────────────────
    st.markdown("### 📉 Training & Validation Loss")
    loss_df = pd.DataFrame({
        "Train Loss": history.history["loss"],
        "Val Loss":   history.history["val_loss"]
    })
    st.line_chart(loss_df)

    st.markdown("### 📋 Classification Report")
    st.dataframe(report_df)

    st.markdown(f"**Accuracy:** {acc:.3f}")


    st.write("Accuracy:", acc)
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

