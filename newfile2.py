import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="NIDS · Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #080f1a !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #cbd5e1;
}

/* Subtle grid background */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(6,182,212,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(6,182,212,0.025) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none; z-index: 0;
}

.block-container {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1200px !important;
    position: relative; z-index: 1;
}

#MainMenu, footer, header { visibility: hidden; }

/* ─── NAV RADIO ─────────────────────────── */
div[role="radiogroup"] {
    gap: 0 !important;
    background: #0d1829 !important;
    border: 1px solid #1e3352 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    display: inline-flex !important;
}
div[role="radiogroup"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 7px 20px !important;
    border-radius: 7px !important;
    color: #475569 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
div[role="radiogroup"] label:has(input:checked) {
    background: #0ea5e9 !important;
    color: #fff !important;
}

/* ─── SECTION HEADERS ───────────────────── */
h1 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.45rem !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.01em !important;
}
h2, h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    margin-top: 1.8rem !important;
    margin-bottom: 0.75rem !important;
}

/* ─── INPUTS ────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #0d1829 !important;
    border: 1px solid #1e3352 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.12) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ─── BUTTON ────────────────────────────── */
.stButton > button {
    background: #0ea5e9 !important;
    color: #fff !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.8rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(14,165,233,0.25) !important;
}
.stButton > button:hover {
    background: #38bdf8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 0 32px rgba(56,189,248,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ─── METRICS ───────────────────────────── */
[data-testid="stMetric"] {
    background: #0d1829 !important;
    border: 1px solid #1e3352 !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.5rem !important;
    color: #0ea5e9 !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: #334155 !important;
}

/* ─── PROGRESS BAR ──────────────────────── */
[data-testid="stProgressBar"] > div {
    background: #0d1829 !important;
    border: 1px solid #1e3352 !important;
    border-radius: 999px !important;
    height: 5px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0369a1, #0ea5e9, #38bdf8) !important;
    border-radius: 999px !important;
}

/* ─── ALERTS ────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ─── FILE UPLOADER ─────────────────────── */
[data-testid="stFileUploader"] {
    background: #0d1829 !important;
    border: 1.5px dashed #1e3352 !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #0ea5e9 !important;
    background: rgba(14,165,233,0.04) !important;
}

/* ─── DATAFRAME ─────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e3352 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ─── DIVIDER ───────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #0f1f35 !important;
    margin: 1.5rem 0 !important;
}

/* ─── SLIDER ────────────────────────────── */
[data-testid="stSlider"] .st-ae { background: #0ea5e9 !important; }
[data-testid="stSlider"] .st-ai { background: #1e3352 !important; }

/* ─── SCROLLBAR ─────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080f1a; }
::-webkit-scrollbar-thumb { background: #1e3352; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #0ea5e9; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def stat_card(label: str, value: str, accent: str = "#0ea5e9") -> str:
    return f"""
    <div style="background:#0d1829; border:1px solid #1e3352;
                border-radius:10px; padding:1rem 1.2rem;
                border-left:3px solid {accent};">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                    color:#334155; text-transform:uppercase; letter-spacing:0.14em;
                    margin-bottom:6px;">{label}</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.35rem;
                    font-weight:600; color:{accent};">{value}</div>
    </div>"""


def section_header(icon: str, title: str) -> None:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px;
                margin:1.8rem 0 0.9rem; padding-bottom:10px;
                border-bottom:1px solid #0f1f35;">
        <span style="font-size:0.95rem;">{icon}</span>
        <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                     font-weight:600; color:#475569;
                     text-transform:uppercase; letter-spacing:0.16em;">{title}</span>
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, color: str = "#0ea5e9") -> str:
    bg = color + "18"
    return (f'<span style="background:{bg}; color:{color}; border:1px solid {color}44;'
            f'border-radius:5px; font-family:IBM Plex Mono,monospace;'
            f'font-size:0.65rem; padding:2px 9px; letter-spacing:0.08em;">{text}</span>')


# ══════════════════════════════════════════════
# LOAD MODEL ARTIFACTS
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    BASE_DIR = os.path.dirname(_file_)
    model = load_model(os.path.join(BASE_DIR, "nids_model.h5"))
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(BASE_DIR, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    return model, scaler, encoders, features


with st.spinner("Loading model…"):
    model, scaler, encoders, features = load_artifacts()


# ══════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between;
            padding:0 0 1.2rem; border-bottom:1px solid #0f1f35; margin-bottom:1rem;">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:36px; height:36px; border-radius:9px;
                    background:rgba(14,165,233,0.12);
                    border:1px solid rgba(14,165,233,0.3);
                    display:flex; align-items:center; justify-content:center;
                    font-size:1rem;">🛡️</div>
        <div>
            <div style="font-family:'IBM Plex Mono',monospace; font-weight:600;
                        font-size:1rem; color:#f1f5f9; letter-spacing:-0.01em;">NIDS</div>
            <div style="font-family:'IBM Plex Sans',sans-serif; font-size:0.72rem;
                        color:#334155; margin-top:1px;">Network Intrusion Detection System</div>
        </div>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
        {badge("KDD Dataset")}
        {badge("Deep Learning", "#22c55e")}
        {badge("v1.0", "#a78bfa")}
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════
menu = st.radio("", ["Home", "Analyze", "Batch Analysis"], horizontal=True,
                label_visibility="collapsed")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════
if menu == "Home":
    st.markdown("""
    <div style="margin-top:1rem; margin-bottom:2rem;">
        <h1 style="font-family:'IBM Plex Mono',monospace; font-size:1.8rem;
                   font-weight:600; color:#f1f5f9; margin-bottom:0.4rem;">
            Network Intrusion<br>
            <span style="color:#0ea5e9;">Detection System</span>
        </h1>
        <p style="color:#475569; font-size:0.9rem; max-width:520px; line-height:1.7;
                  font-family:'IBM Plex Sans',sans-serif;">
            Deep learning model trained on the KDD Cup dataset.
            Detect malicious network traffic in real-time or batch mode.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Model", "Deep NN", "#0ea5e9"),
        ("Dataset", "KDD '99", "#22c55e"),
        ("Accuracy", "~99%", "#a78bfa"),
        ("Classes", "Normal / Attack", "#f59e0b"),
    ]
    for col, (label, val, color) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(stat_card(label, val, color), unsafe_allow_html=True)

    section_header("📋", "How It Works")
    steps_html = """
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-top:0.5rem;">
    """
    steps = [
        ("01", "Input Traffic", "Enter network connection features manually or upload a CSV batch file."),
        ("02", "Feature Scaling", "Features are normalised using the pre-trained StandardScaler."),
        ("03", "Inference", "The deep neural network outputs an attack probability score."),
    ]
    for num, title, desc in steps:
        steps_html += f"""
        <div style="background:#0d1829; border:1px solid #1e3352; border-radius:10px;
                    padding:1.1rem 1.2rem;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem;
                        color:#1e3352; margin-bottom:8px;">{num}</div>
            <div style="font-weight:600; font-size:0.9rem; color:#e2e8f0;
                        margin-bottom:6px;">{title}</div>
            <div style="font-size:0.82rem; color:#475569; line-height:1.6;">{desc}</div>
        </div>"""
    steps_html += "</div>"
    st.markdown(steps_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# ANALYZE — SINGLE INPUT
# ══════════════════════════════════════════════
elif menu == "Analyze":
    section_header("🔍", "Analyze Network Traffic")

    st.markdown("""
    <p style="color:#475569; font-size:0.85rem; margin-bottom:1.5rem;">
        Enter connection attributes below to classify as normal or attack traffic.
    </p>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("##### Connection Attributes")
        protocol = st.selectbox("Protocol Type", encoders["protocol_type"].classes_)
        service  = st.selectbox("Service", encoders["service"].classes_)
        flag     = st.selectbox("Connection Flag", encoders["flag"].classes_)

    with right:
        st.markdown("##### Traffic Metrics")
        src_bytes  = st.number_input("Source Bytes",      min_value=0, value=232)
        dst_bytes  = st.number_input("Destination Bytes", min_value=0, value=8153)
        count      = st.number_input("Connection Count",  min_value=0, value=10)
        srv_count  = st.number_input("Service Count",     min_value=0, value=10)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run = st.button("⚡  Run Analysis")

    if run:
        row = dict.fromkeys(features, 0)
        row["protocol_type"] = encoders["protocol_type"].transform([protocol])[0]
        row["service"]       = encoders["service"].transform([service])[0]
        row["flag"]          = encoders["flag"].transform([flag])[0]
        row["src_bytes"]     = src_bytes
        row["dst_bytes"]     = dst_bytes
        row["count"]         = count
        row["srv_count"]     = srv_count
        row.update({
            "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0,
            "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
            "dst_host_count": 255, "dst_host_srv_count": 255,
        })

        df_in   = pd.DataFrame([row])[features]
        df_in   = df_in.apply(pd.to_numeric, errors="coerce").fillna(0)
        df_sc   = scaler.transform(df_in)
        prob    = model.predict(df_sc)[0][0]
        is_atk  = prob > 0.2
        label   = "ATTACK" if is_atk else "NORMAL"
        conf    = prob if is_atk else 1 - prob

        st.markdown("<hr>", unsafe_allow_html=True)
        section_header("🎯", "Prediction Result")

        verdict_color = "#ef4444" if is_atk else "#22c55e"
        verdict_icon  = "⚠" if is_atk else "✓"
        verdict_bg    = "rgba(239,68,68,0.06)" if is_atk else "rgba(34,197,94,0.06)"
        verdict_border= "rgba(239,68,68,0.25)" if is_atk else "rgba(34,197,94,0.25)"

        st.markdown(f"""
        <div style="background:{verdict_bg}; border:1px solid {verdict_border};
                    border-left:3px solid {verdict_color};
                    border-radius:10px; padding:1.2rem 1.4rem;
                    display:flex; align-items:center; justify-content:space-between;
                    margin-bottom:1.2rem;">
            <div style="display:flex; align-items:center; gap:14px;">
                <span style="font-size:1.5rem; color:{verdict_color};">{verdict_icon}</span>
                <div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:1.2rem;
                                font-weight:600; color:{verdict_color};">{label}</div>
                    <div style="font-size:0.78rem; color:#475569; margin-top:3px;">
                        {'Malicious activity detected in this connection' if is_atk else 'No threats detected — connection appears normal'}
                    </div>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:1.6rem;
                            font-weight:600; color:{verdict_color};">{conf*100:.1f}%</div>
                <div style="font-size:0.65rem; color:#334155;
                            text-transform:uppercase; letter-spacing:0.12em;">confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Attack Probability", f"{prob:.4f}")
        with m2: st.metric("Normal Score", f"{1-prob:.4f}")
        with m3: st.metric("Threshold", "0.200")

        st.progress(int(conf * 100))


# ══════════════════════════════════════════════
# BATCH ANALYSIS
# ══════════════════════════════════════════════
elif menu == "Batch Analysis":
    section_header("📂", "Batch CSV Analysis")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"],
                                     label_visibility="collapsed")
    if uploaded_file is None:
        st.markdown("""
        <div style="background:#0d1829; border:1.5px dashed #1e3352; border-radius:12px;
                    padding:2.5rem; text-align:center; margin-top:0.5rem;">
            <div style="font-size:1.5rem; margin-bottom:8px;">📁</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                        color:#334155; text-transform:uppercase; letter-spacing:0.1em;">
                Drop your CSV file above to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    try:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # ── Overview stats ──────────────────────
    section_header("📊", "Dataset Overview")

    o1, o2, o3, o4 = st.columns(4)
    with o1: st.markdown(stat_card("Rows", f"{len(df):,}"), unsafe_allow_html=True)
    with o2: st.markdown(stat_card("Columns", str(len(df.columns))), unsafe_allow_html=True)
    with o3: st.markdown(stat_card("Numeric cols", str(len(df.select_dtypes(include='number').columns)), "#22c55e"), unsafe_allow_html=True)
    with o4: st.markdown(stat_card("Missing values", str(df.isnull().sum().sum()), "#f59e0b"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("View full statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ── Filtering ───────────────────────────
    section_header("🧪", "Filter Data")

    fc1, fc2 = st.columns(2)
    with fc1:
        filter_col = st.selectbox("Column", df.columns.tolist(), key="fcol")
    with fc2:
        filter_val = st.selectbox("Value", df[filter_col].dropna().unique(), key="fval")

    filtered_df = df[df[filter_col] == filter_val]
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#334155;
                margin:0.4rem 0 0.6rem;">
        {len(filtered_df):,} rows matched
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

    # ── Plotting ────────────────────────────
    section_header("📈", "Plot Data")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 2:
        pc1, pc2, pc3 = st.columns([1, 1, 0.7])
        with pc1: x_col = st.selectbox("X-axis", numeric_cols, key="xc")
        with pc2: y_col = st.selectbox("Y-axis", numeric_cols, key="yc")
        with pc3:
            plot_type = st.radio("Type", ["Line", "Pie"], horizontal=True)

        if st.button("Generate Plot"):
            plot_df = df[[x_col, y_col]].dropna()
            fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0d1829")
            ax.set_facecolor("#0d1829")

            if plot_type == "Line":
                ax.plot(plot_df[x_col], plot_df[y_col],
                        color="#0ea5e9", linewidth=1.5, alpha=0.9)
                ax.fill_between(plot_df[x_col], plot_df[y_col],
                                color="#0ea5e9", alpha=0.06)
                ax.set_xlabel(x_col, color="#475569", fontsize=9)
                ax.set_ylabel(y_col, color="#475569", fontsize=9)
            else:
                agg = plot_df.groupby(x_col)[y_col].sum()
                colors = ["#0ea5e9", "#22c55e", "#a78bfa", "#f59e0b",
                          "#ef4444", "#38bdf8", "#34d399"]
                ax.pie(agg, labels=agg.index, autopct="%1.1f%%",
                       colors=colors[:len(agg)],
                       textprops={"color": "#94a3b8", "fontsize": 8})

            for spine in ax.spines.values():
                spine.set_edgecolor("#1e3352")
            ax.tick_params(colors="#475569", labelsize=8)
            ax.grid(color="#0f1f35", linewidth=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning("Need at least two numeric columns for plotting.")

    # ── Classification ──────────────────────
    section_header("🤖", "Train & Evaluate Classifier")

    all_cols = df.columns.tolist()
    cl1, cl2 = st.columns(2)
    with cl1:
        target_col = st.selectbox("Target column", all_cols)
    with cl2:
        feature_cols = st.multiselect(
            "Feature columns",
            [c for c in all_cols if c != target_col],
            default=[c for c in numeric_cols if c != target_col][:4],
        )

    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, step=0.05,
                          help="Fraction of data held out for evaluation")

    if not feature_cols:
        st.warning("Select at least one feature column.")
        st.stop()

    if st.button("Train & Evaluate"):
        model_df = df[[target_col] + feature_cols].dropna().copy()
        clf_encoders = {}
        for col in [target_col] + feature_cols:
            if not pd.api.types.is_numeric_dtype(model_df[col]):
                le = LabelEncoder()
                model_df[col] = le.fit_transform(model_df[col].astype(str))
                clf_encoders[col] = le

        X, y = model_df[feature_cols], model_df[target_col]
        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        pipeline = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42))
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report_df = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        ).transpose()
        cm = confusion_matrix(y_test, y_pred)

        section_header("📋", "Results")

        r1, r2, r3 = st.columns(3)
        with r1: st.markdown(stat_card("Accuracy",  f"{acc*100:.2f}%", "#22c55e"), unsafe_allow_html=True)
        with r2: st.markdown(stat_card("Train set", f"{len(X_train):,}", "#0ea5e9"), unsafe_allow_html=True)
        with r3: st.markdown(stat_card("Test set",  f"{len(X_test):,}", "#a78bfa"), unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.dataframe(report_df.style.format(precision=3), use_container_width=True)

        # Confusion matrix heatmap
        section_header("🔲", "Confusion Matrix")
        fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor="#0d1829")
        ax2.set_facecolor("#0d1829")
        im = ax2.imshow(cm, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax2).ax.tick_params(colors="#475569", labelsize=8)
        ax2.set_xlabel("Predicted", color="#475569", fontsize=9)
        ax2.set_ylabel("Actual", color="#475569", fontsize=9)
        ax2.tick_params(colors="#475569", labelsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="#f1f5f9", fontsize=10, fontweight="bold")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#1e3352")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

