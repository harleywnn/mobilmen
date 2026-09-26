"""
Aplikasi Streamlit — Segmentasi Mobil Bekas (K-Means Clustering)
Deployment dari project CRISP-DM

Jalankan dengan:
    py -m streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Mobil Bekas | K-Means",
    page_icon="🚗",
    layout="wide",
)

CURRENT_YEAR = 2026

CLUSTER_LABELS = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
}

# ------------------------------------------------------------------
# Loader (cache supaya tidak reload berulang setiap interaksi)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("kmeans_model.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    return scaler, model, feature_columns


scaler, model, feature_columns = load_artifacts()

# ------------------------------------------------------------------
# Fungsi Prediksi
# ------------------------------------------------------------------
def build_features(row: pd.DataFrame) -> pd.DataFrame:
    row = row.copy()
    row["Car_Age"] = CURRENT_YEAR - row["Model_Year"]
    
    # One-hot encoding untuk variabel kategorikal
    row_encoded = pd.get_dummies(row[["Fuel_Type", "Transmission"]], drop_first=True)
    
    # Gabungkan dengan fitur numerik
    row_features = pd.concat(
        [row[["Car_Age", "Engine_Size", "Mileage", "Horsepower", "Price"]], row_encoded],
        axis=1,
    )
    
    # Samakan kolom dengan training data
    row_features = row_features.reindex(columns=feature_columns, fill_value=0)
    return row_features


def predict_cluster(row: pd.DataFrame):
    features = build_features(row)
    scaled = scaler.transform(features)
    cluster = model.predict(scaled)[0]
    return int(cluster)


# ------------------------------------------------------------------
# Sidebar Navigasi
# ------------------------------------------------------------------
st.sidebar.title("🚗 Navigasi")
page = st.sidebar.radio(
    "Pilih halaman",
    ["Prediksi Mobil Baru", "Prediksi Massal (CSV)"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model K-Means dilatih menggunakan metodologi CRISP-DM."
)

# ==================================================================
# HALAMAN 1: PREDIKSI SATU MOBIL BARU
# ==================================================================
if page == "Prediksi Mobil Baru":
    st.title("Prediksi Cluster untuk Mobil Baru")
    st.markdown("Masukkan spesifikasi mobil, lalu sistem akan menentukan cluster/segmennya.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            brand = st.text_input("Brand", value="Toyota")
            model_year = st.number_input("Tahun Produksi (Model Year)", min_value=2000, max_value=CURRENT_YEAR, value=2020)
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
        with c2:
            engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
            mileage = st.number_input("Mileage (km)", min_value=0, max_value=400000, value=40000, step=1000)
            transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
        with c3:
            horsepower = st.number_input("Horsepower", min_value=50, max_value=600, value=200)
            price = st.number_input("Price ($)", min_value=1000, max_value=200000, value=25000, step=500)

        submitted = st.form_submit_button("Prediksi Cluster")

    if submitted:
        input_row = pd.DataFrame([{
            "Model_Year": model_year,
            "Engine_Size": engine_size,
            "Mileage": mileage,
            "Horsepower": horsepower,
            "Price": price,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
        }])

        cluster = predict_cluster(input_row)
        label = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")

        st.success(f"Mobil ini diprediksi masuk ke **{label}**")

# ==================================================================
# HALAMAN 2: PREDIKSI MASSAL DARI FILE CSV
# ==================================================================
elif page == "Prediksi Massal (CSV)":
    st.title("Prediksi Cluster secara Massal")
    st.markdown(
        "Unggah file CSV berisi data mobil baru dengan kolom berikut: "
        "`Model_Year, Engine_Size, Mileage, Horsepower, Price, Fuel_Type, Transmission`"
    )

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        required_cols = ["Model_Year", "Engine_Size", "Mileage", "Horsepower", "Price", "Fuel_Type", "Transmission"]
        missing = [c for c in required_cols if c not in new_data.columns]

        if missing:
            st.error(f"Kolom berikut tidak ditemukan pada file CSV: {missing}")
        else:
            features = build_features(new_data)
            scaled = scaler.transform(features)
            clusters = model.predict(scaled)
            new_data["Predicted_Cluster"] = clusters
            new_data["Predicted_Label"] = new_data["Predicted_Cluster"].map(CLUSTER_LABELS)

            st.success(f"Berhasil memprediksi cluster untuk {len(new_data)} baris data.")
            st.dataframe(new_data, use_container_width=True)

            csv_out = new_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh Hasil Prediksi (CSV)",
                data=csv_out,
                file_name="hasil_prediksi_cluster.csv",
                mime="text/csv",
            )