import streamlit as st
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import base64
import os
from utilities.Analyser import run_analysis
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from utilities.ref_doc_gen import answer_generation, save_to_word_advanced
from dotenv import load_dotenv
load_dotenv()
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="StackEMS.ai", layout="wide")

# =========================
# BACKGROUND LOGO (SAFE)
# =========================
def add_bg_logo(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, #0f172a, #1e293b);
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 50%;
        left: 50%;
        width: 500px;
        height: 500px;
        transform: translate(-50%, -50%);
        background-image: url("data:image/png;base64,{encoded}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        opacity: 0.07;
        filter: blur(6px);
        z-index: -1;
        pointer-events: none;
    }}
    </style>
    """, unsafe_allow_html=True)

# add_bg_logo("StackEMS.jpeg")

# =========================
# PREMIUM CSS
# =========================
st.markdown("""
<style>

.header-container {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 15px 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(8px);
    margin-bottom: 20px;
}

.logo-box img {
    width: 110px;
    border-radius: 12px;
}

.title-box h1 {
    margin: 0;
    font-size: 40px;
    color: #e2e8f0;
}

.title-box p {
    margin: 0;
    color: gray;
}

.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
with open("utilities/StackEMS.jpeg", "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div class="header-container">
    <div class="logo-box">
        <img src="data:image/png;base64,{logo_base64}" />
    </div>
    <div class="title-box">
        <h1>StackEMS.ai</h1>
        <p>Automated Emissions Monitoring & AI Audit System</p>
    </div>
</div>
""", unsafe_allow_html=True)

api_key = os.getenv("GROQ_API_KEY")

# =========================
# PREPROCESS
# =========================
def preprocess(df):
    df = df.rename(columns={
        "NOx (mg/Nm3)": "NOx",
        "SO2 (mg/Nm3)": "SO2",
        "PM (mg/Nm3)": "PM",
        "Temp (°C)": "Temp",
        "Flow (Nm3/hr)": "Flow",
        "O2 (%)": "O2"
    })

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Stack", "Timestamp"])

    df["group"] = df["Stack"].astype(str)
    df["time_idx"] = df.groupby("group").cumcount()

    df = df[df["Stack"] == "Kiln"]
    df = df.ffill().bfill()

    df["hour"] = df["Timestamp"].dt.hour
    df["day"] = df["Timestamp"].dt.day

    return df

# =========================
# MODEL RUNNER
# =========================
def run_model(df, target, training_file, model_file):

    with open(training_file, "rb") as f:
        training = pickle.load(f)

    model = TemporalFusionTransformer.load_from_checkpoint(model_file)
    model.eval()

    df_new = df.copy()

    for lag in [6, 12, 24]:
        df_new[f"{target}_lag_{lag}"] = df_new[f"{target}"].shift(lag)
        df_new[f"flow_lag_{lag}"] = df_new["Flow"].shift(lag)

    for lag in [6]:
        df_new[f"temp_lag_{lag}"] = df_new["Temp"].shift(lag)

    # ROLLING
    df_new[f"{target}_roll_mean_6"] = df_new[f"{target}"].rolling(6).mean()
    df_new[f"{target}_roll_std_6"] = df_new[f"{target}"].rolling(6).std()

    df_new[f"flow_roll_std_6"] = df_new["Flow"].rolling(6).std()
    df_new["temp_roll_std_6"] = df_new["Temp"].rolling(6).std()
    df_new["flow_temp_interaction"] = df_new["Flow"] * df_new["Temp"]
    df_new["weight"] = 1 + (df_new[f"{target}"] / df_new[f"{target}"].mean())

    df_new = df_new.dropna()

    dataset = TimeSeriesDataSet.from_dataset(
        training, df_new, predict=True, stop_randomization=True
    )

    loader = dataset.to_dataloader(train=False, batch_size=64, num_workers=4)

    preds = model.predict(loader, mode="raw", return_x=True)

    raw = preds.output
    x = preds.x

    pred = raw["prediction"][:, :, raw["prediction"].shape[2] // 2].detach().cpu().numpy().flatten()
    actual = x["decoder_target"].detach().cpu().numpy().flatten()

    return model, x, raw, actual, pred

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload CEMS Data", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df = preprocess(df)

    if st.button("🚀 Run Predictions"):
        st.session_state["run"] = True

    if st.session_state.get("run", False):

        if "data" not in st.session_state:

            with st.spinner("Running models..."):

                nox = run_model(df, "NOx", "models/Nx_training_dataset.pkl", "models/tft_model_NOx.ckpt")
                so2 = run_model(df, "SO2", "models/training_dataset_SO2.pkl", "models/tft_model_SO2.ckpt")
                pm  = run_model(df, "PM", "models/training_dataset_PM.pkl", "models/tft_model_PM.ckpt")

                st.session_state["data"] = {"nox": nox, "so2": so2, "pm": pm}

        data = st.session_state["data"]

        tab1, tab2, tab3 = st.tabs(["NOx", "SO2", "PM"])

        def plot_section(model, x, raw):
            plt.figure(figsize=(5,3))
            model.plot_prediction(x, raw, idx=0)
            st.pyplot(plt.gcf(),width=600)
            plt.clf()

        with tab1:
            plot_section(*data["nox"][:3])
        with tab2:
            plot_section(*data["so2"][:3])
        with tab3:
            plot_section(*data["pm"][:3])

        # =========================
        # OUTPUT
        # =========================
        output_df = pd.DataFrame({
            "NOx": data["nox"][3],
            "SO2": data["so2"][3],
            "PM": data["pm"][3],
        })

        # =========================
        # AI ANALYSIS
        # =========================
        st.markdown("## 🧠 AI Insights")

        if st.button("Run Analysis"):
            st.session_state["analysis"] = True

        if st.session_state.get("analysis", False):
            analysis = run_analysis(output_df,api_key)

            col1, col2 = st.columns(2)

            with col1:
                for s in analysis.get("Suggestions", []):
                    st.info(s)

            with col2:
                for a in analysis.get("Immediate Actionables", []):
                    st.error(a)

        # =========================
        # REPORT GENERATOR
        # =========================
        st.markdown("## 📄 Audit Report")

        report_file = st.file_uploader("Upload Monthly File", key="report")
        template = st.text_area("Report Template")

        if st.button("Generate Report"):

            if not report_file or not template or not api_key:
                print(api_key)
                st.error("Missing inputs")
            else:
                df_r = pd.read_excel(report_file)

                text = answer_generation(df_r.head(300), template, api_key)
                file = "report.docx"

                save_to_word_advanced(text, file)

                with open(file, "rb") as f:
                    st.download_button("Download Report", f, file)

        # =========================
        # DOWNLOAD
        # =========================
        st.download_button(
            "⬇ Download Predictions",
            output_df.to_csv(index=False),
            "predictions.csv"
        )

        if st.button("🔄 Reset"):
            st.session_state.clear()