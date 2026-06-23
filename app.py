import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_sdn.csv"
MODEL_DIR = BASE_DIR / "exported_lsh_knn_model"

st.set_page_config(
    page_title="DDoS SDN Detection",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style():
    st.markdown(
        """
        <style>
            :root {
                --navy: #0b1f3a;
                --navy-soft: #12365f;
                --blue: #1d4ed8;
                --sky: #eaf2ff;
                --border: #d7e3f5;
                --text: #102033;
            }

            .stApp {
                background: #ffffff;
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b1f3a 0%, #12365f 100%);
            }

            [data-testid="stSidebar"] * {
                color: #ffffff !important;
            }

            [data-testid="stSidebar"] .stRadio label {
                padding: 8px 10px;
                border-radius: 8px;
            }

            .hero {
                background: linear-gradient(135deg, #0b1f3a 0%, #12365f 58%, #1d4ed8 100%);
                color: white;
                padding: 28px 32px;
                border-radius: 8px;
                margin-bottom: 18px;
            }

            .hero h1 {
                margin: 0 0 8px 0;
                font-size: 34px;
                letter-spacing: 0;
            }

            .hero p {
                margin: 0;
                color: #dbeafe;
                font-size: 16px;
            }

            .metric-card {
                border: 1px solid var(--border);
                border-left: 5px solid var(--blue);
                border-radius: 8px;
                padding: 18px 18px 14px 18px;
                background: #ffffff;
                box-shadow: 0 8px 24px rgba(11, 31, 58, 0.06);
                min-height: 112px;
            }

            .metric-label {
                color: #52657d;
                font-size: 13px;
                font-weight: 700;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .metric-value {
                color: var(--navy);
                font-size: 30px;
                font-weight: 800;
                line-height: 1;
            }

            .section-title {
                color: var(--navy);
                font-size: 21px;
                font-weight: 800;
                margin: 12px 0 6px 0;
            }

            .prediction-box {
                border-radius: 8px;
                padding: 22px;
                border: 1px solid var(--border);
                background: var(--sky);
            }

            .normal {
                border-left: 6px solid #16a34a;
                background: #f0fdf4;
            }

            .attack {
                border-left: 6px solid #dc2626;
                background: #fef2f2;
            }

            .prediction-confidence {
                margin: 8px 0 0 0;
                color: #52657d;
                font-size: 14px;
                font-weight: 600;
            }

            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 14px;
            }

            .stButton > button {
                background: var(--navy);
                color: white;
                border: 1px solid var(--navy);
                border-radius: 8px;
                font-weight: 700;
            }

            .stButton > button:hover {
                background: var(--blue);
                border-color: var(--blue);
                color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_metadata():
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_dataset_preview():
    return pd.read_csv(DATA_PATH)


def vector_values(value, size=None):
    values = np.asarray(value["values"], dtype=float)
    indices = value.get("indices")
    vector_size = value.get("size") or size
    if indices is None:
        return values

    dense = np.zeros(int(vector_size), dtype=float)
    dense[np.asarray(indices, dtype=int)] = values
    return dense


@st.cache_resource(show_spinner=False)
def load_fast_assets():
    metadata = load_metadata()
    feature_size = len(metadata["feature_cols"])
    train_df = pd.read_parquet(MODEL_DIR / "train_hashed")
    train_features = np.vstack(train_df["features"].map(lambda item: vector_values(item, feature_size)).to_numpy())
    train_labels = train_df["label"].astype(int).to_numpy()

    scaler_df = pd.read_parquet(MODEL_DIR / "scaler_model" / "data")
    scaler_std = vector_values(scaler_df.iloc[0]["std"])
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)

    index_maps = {}
    stage_files = sorted(
        glob.glob(str(MODEL_DIR / "encoding_model" / "stages" / "*" / "data" / "*.parquet"))
    )
    for column, stage_file in zip(metadata["categorical_cols"], stage_files):
        labels = pd.read_parquet(stage_file).iloc[0]["labelsArray"][0]
        index_maps[column] = {str(label): float(index) for index, label in enumerate(labels)}

    majority_label = int(pd.Series(train_labels).mode().iloc[0])
    return train_features, train_labels, scaler_std, index_maps, majority_label


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def label_name(label):
    return "Attack / DDoS" if int(label) == 1 else "Normal"


def label_class(label):
    return "attack" if int(label) == 1 else "normal"


def dashboard(metadata, df):
    st.markdown(
        """
        <div class="hero">
            <h1>DDoS SDN Detection</h1>
            <p>Dashboard monitoring untuk model Spark LSH-KNN pada lalu lintas Software Defined Network.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = metadata["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    with c2:
        metric_card("Precision", f"{metrics['precision'] * 100:.1f}%")
    with c3:
        metric_card("Recall", f"{metrics['recall'] * 100:.1f}%")
    with c4:
        metric_card("F1 Score", f"{metrics['f1_score'] * 100:.1f}%")

    st.markdown('<div class="section-title">Ringkasan Data</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Record", f"{len(df):,}")
    d2.metric("Jumlah Fitur Model", len(metadata["feature_cols"]))
    d3.metric("Data Normal", f"{int((df['label'] == 0).sum()):,}")
    d4.metric("Data Attack", f"{int((df['label'] == 1).sum()):,}")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="section-title">Distribusi Label</div>', unsafe_allow_html=True)
        label_counts = (
            df["label"]
            .map({0: "Normal", 1: "Attack / DDoS"})
            .value_counts()
            .rename_axis("Label")
            .reset_index(name="Jumlah")
        )
        st.bar_chart(label_counts, x="Label", y="Jumlah", color="#1d4ed8", height=300)

    with right:
        st.markdown('<div class="section-title">Parameter Model</div>', unsafe_allow_html=True)
        params = metadata["parameters"]
        st.dataframe(
            pd.DataFrame(
                [
                    {"Parameter": "K Nearest Neighbors", "Nilai": params["k"]},
                    {"Parameter": "Bucket Length", "Nilai": params["bucket_length"]},
                    {"Parameter": "Num Hash Tables", "Nilai": params["num_hash_tables"]},
                    {"Parameter": "Distance Threshold", "Nilai": params["distance_threshold"]},
                    {"Parameter": "LSH Train Fraction", "Nilai": params["lsh_train_fraction"]},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown('<div class="section-title">Protocol Berdasarkan Label</div>', unsafe_allow_html=True)
        protocol_summary = (
            df.assign(label_name=df["label"].map({0: "Normal", 1: "Attack / DDoS"}))
            .groupby(["Protocol", "label_name"])
            .size()
            .reset_index(name="Jumlah")
        )
        st.bar_chart(
            protocol_summary,
            x="Protocol",
            y="Jumlah",
            color="label_name",
            height=300,
        )

    with chart_right:
        st.markdown('<div class="section-title">Top Source IP Attack</div>', unsafe_allow_html=True)
        top_attack_src = (
            df[df["label"] == 1]
            .groupby("src")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .rename_axis("Source IP")
            .reset_index(name="Jumlah Attack")
        )
        st.bar_chart(
            top_attack_src,
            x="Source IP",
            y="Jumlah Attack",
            color="#1d4ed8",
            height=300,
        )

    st.markdown('<div class="section-title">Rata-Rata Trafik Berdasarkan Label</div>', unsafe_allow_html=True)
    traffic_cols = ["pktcount", "bytecount", "flows", "packetins", "pktrate", "tot_kbps"]
    traffic_summary = df.groupby("label")[traffic_cols].mean().rename(index={0: "Normal", 1: "Attack / DDoS"})
    st.dataframe(traffic_summary.round(2), use_container_width=True)

    st.markdown('<div class="section-title">Contoh Dataset</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), hide_index=True, use_container_width=True)


def build_default_values(df, mode):
    if mode == "Auto dari Dataset":
        row = df.sample(1, random_state=None).iloc[0].to_dict()
    else:
        row = df[df["label"] == 0].iloc[0].to_dict()
    return row


def build_label_sample(df, label):
    sample_df = df[df["label"] == label]
    if sample_df.empty:
        return df.sample(1, random_state=None).iloc[0].to_dict()
    return sample_df.sample(1, random_state=None).iloc[0].to_dict()


def apply_sample_to_widgets(sample, metadata):
    st.session_state.sample_values = sample
    for column in [*metadata["categorical_cols"], *metadata["numeric_cols"]]:
        if column in sample:
            st.session_state[column] = sample[column]


def render_input_form(metadata, df):
    input_mode = st.radio(
        "Mode Input",
        ["Auto dari Dataset", "Manual"],
        horizontal=True,
        key="input_mode",
    )

    action_cols = st.columns(3)
    with action_cols[0]:
        if st.button("Contoh Normal", use_container_width=True):
            apply_sample_to_widgets(build_label_sample(df, 0), metadata)
    with action_cols[1]:
        if st.button("Contoh DDoS", use_container_width=True):
            apply_sample_to_widgets(build_label_sample(df, 1), metadata)
    with action_cols[2]:
        if st.button("Acak Dataset", use_container_width=True):
            apply_sample_to_widgets(build_default_values(df, "Auto dari Dataset"), metadata)

    if "sample_values" not in st.session_state:
        apply_sample_to_widgets(build_default_values(df, input_mode), metadata)

    if input_mode == "Manual":
        values = st.session_state.sample_values.copy()
    else:
        values = st.session_state.sample_values.copy()

    categorical_cols = metadata["categorical_cols"]
    numeric_cols = metadata["numeric_cols"]

    st.markdown('<div class="section-title">Identitas Trafik</div>', unsafe_allow_html=True)
    id_cols = st.columns(3)
    with id_cols[0]:
        src_options = sorted(df["src"].dropna().unique().tolist())
        if st.session_state.get("src") not in src_options:
            st.session_state["src"] = values.get("src", src_options[0])
        values["src"] = st.selectbox("Source IP", src_options, key="src")
    with id_cols[1]:
        dst_options = sorted(df["dst"].dropna().unique().tolist())
        if st.session_state.get("dst") not in dst_options:
            st.session_state["dst"] = values.get("dst", dst_options[0])
        values["dst"] = st.selectbox("Destination IP", dst_options, key="dst")
    with id_cols[2]:
        protocol_options = sorted(df["Protocol"].dropna().unique().tolist())
        if st.session_state.get("Protocol") not in protocol_options:
            st.session_state["Protocol"] = values.get("Protocol", protocol_options[0])
        values["Protocol"] = st.selectbox("Protocol", protocol_options, key="Protocol")

    st.markdown('<div class="section-title">Fitur Numerik</div>', unsafe_allow_html=True)
    for group_start in range(0, len(numeric_cols), 3):
        cols = st.columns(3)
        for col, feature in zip(cols, numeric_cols[group_start : group_start + 3]):
            min_value = float(df[feature].min())
            max_value = float(df[feature].max())
            current = float(values.get(feature, df[feature].median()))
            step = max((max_value - min_value) / 1000, 1.0)
            if feature not in st.session_state:
                st.session_state[feature] = max(min(current, max_value), min_value)
            with col:
                values[feature] = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    step=step,
                    format="%.4f",
                    key=feature,
                )

    return {col: values[col] for col in [*categorical_cols, *numeric_cols]}


def predict(input_values, metadata):
    train_features, train_labels, scaler_std, index_maps, majority_label = load_fast_assets()
    params = metadata["parameters"]

    feature_values = []
    for column in metadata["feature_cols"]:
        if column.endswith("_index"):
            raw_column = column.replace("_index", "")
            mapping = index_maps.get(raw_column, {})
            fallback_index = float(len(mapping))
            feature_values.append(mapping.get(str(input_values.get(raw_column, "")), fallback_index))
        else:
            feature_values.append(float(input_values[column]))

    scaled_features = np.asarray(feature_values, dtype=float) / scaler_std
    distances = np.linalg.norm(train_features - scaled_features, axis=1)
    k = int(params["k"])
    nearest_indices = np.argpartition(distances, k - 1)[:k]
    nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]

    if len(nearest_indices) == 0:
        predicted_label = majority_label
        confidence = 0.0
        neighbor_view = pd.DataFrame(columns=["label", "distance"])
    else:
        neighbor_view = pd.DataFrame(
            {
                "label": train_labels[nearest_indices].astype(int),
                "distance": distances[nearest_indices].astype(float),
            }
        )
        votes = neighbor_view["label"].astype(int).value_counts()
        predicted_label = int(votes.idxmax())
        confidence = float(votes.max() / len(neighbor_view))

    return predicted_label, confidence, neighbor_view


def prediction_page(metadata, df):
    st.markdown(
        """
        <div class="hero">
            <h1>Prediksi Trafik SDN</h1>
            <p>Masukkan fitur trafik secara manual atau gunakan auto input dari dataset untuk memprediksi Normal atau Attack.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_values = render_input_form(metadata, df)

    st.markdown("")
    if st.button("Prediksi Sekarang", use_container_width=True):
        status = st.empty()
        progress = st.progress(12, text="Menyiapkan input trafik...")
        with st.spinner("Prediksi sedang diproses..."):
            progress.progress(45, text="Memuat model KNN lokal...")
            status.info("Model akan tersimpan di cache setelah pemakaian pertama, jadi prediksi berikutnya lebih cepat.")
            predicted_label, confidence, neighbors = predict(input_values, metadata)
            progress.progress(100, text="Prediksi selesai.")
            status.empty()

        result_class = label_class(predicted_label)
        display_confidence = min(confidence * 100, 99.0)
        st.markdown(
            f"""
            <div class="prediction-box {result_class}">
                <div class="metric-label">Hasil Prediksi</div>
                <div class="metric-value">{label_name(predicted_label)}</div>
                <p class="prediction-confidence">Confidence: {display_confidence:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Tetangga Terdekat</div>', unsafe_allow_html=True)
        neighbor_view = neighbors.copy()
        neighbor_view["label_name"] = neighbor_view["label"].astype(int).map(label_name)
        st.dataframe(neighbor_view, hide_index=True, use_container_width=True)


def main():
    inject_style()
    metadata = load_metadata()
    df = load_dataset_preview()

    st.sidebar.markdown("## DDoS SDN")
    st.sidebar.caption("Spark LSH-KNN Detection")
    page = st.sidebar.radio("Menu", ["Dashboard", "Prediksi"], label_visibility="collapsed")

    if page == "Dashboard":
        dashboard(metadata, df)
    else:
        prediction_page(metadata, df)


if __name__ == "__main__":
    main()
