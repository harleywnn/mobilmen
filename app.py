"""
Aplikasi Streamlit — Segmentasi Mobil Bekas (K-Means Clustering)
Deployment dari project CRISP-DM: car_price_dataset.csv

Jalankan dengan:
    streamlit run app.py

File pendukung yang wajib berada di folder yang sama:
    - scaler.joblib
    - kmeans_model.joblib
    - feature_columns.joblib
    - pca_model.joblib
    - car_price_dataset_clustered.csv
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
# Ganti value di dictionary di atas dengan nama segmen hasil interpretasi bisnismu,
# misal: {0: "Ekonomis", 1: "Menengah", 2: "Premium", 3: "Performa Tinggi"}


# ------------------------------------------------------------------
# Loader (cache supaya tidak reload berulang setiap interaksi)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("kmeans_model.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    pca = joblib.load("pca_model.joblib")
    return scaler, model, feature_columns, pca


@st.cache_data
def load_data():
    df = pd.read_csv("car_price_dataset_clustered.csv")
    df["Cluster_Label"] = df["Cluster"].map(CLUSTER_LABELS)
    return df


scaler, model, feature_columns, pca = load_artifacts()
df = load_data()


# ------------------------------------------------------------------
# Fungsi prediksi
# ------------------------------------------------------------------
def build_features(row: pd.DataFrame) -> pd.DataFrame:
    row = row.copy()
    row["Car_Age"] = CURRENT_YEAR - row["Model_Year"]
    row_encoded = pd.get_dummies(row[["Fuel_Type", "Transmission"]], drop_first=True)
    row_features = pd.concat(
        [row[["Car_Age", "Engine_Size", "Mileage", "Horsepower", "Price"]], row_encoded],
        axis=1,
    )
    row_features = row_features.reindex(columns=feature_columns, fill_value=0)
    return row_features


def predict_cluster(row: pd.DataFrame):
    features = build_features(row)
    scaled = scaler.transform(features)
    cluster = model.predict(scaled)[0]
    pca_coords = pca.transform(scaled)[0]
    return int(cluster), pca_coords


# ------------------------------------------------------------------
# Sidebar navigasi
# ------------------------------------------------------------------
st.sidebar.title("🚗 Navigasi")
page = st.sidebar.radio(
    "Pilih halaman",
    ["Ringkasan", "Eksplorasi Cluster", "Prediksi Mobil Baru", "Prediksi Massal (CSV)"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model K-Means dilatih pada dataset `car_price_dataset.csv` "
    "mengikuti metodologi CRISP-DM."
)

# ==================================================================
# HALAMAN 1: RINGKASAN
# ==================================================================
if page == "Ringkasan":
    st.title("Segmentasi Mobil Bekas Berdasarkan K-Means Clustering")
    st.markdown(
        "Aplikasi ini adalah hasil **deployment** dari project clustering yang mengikuti "
        "metodologi CRISP-DM. Model mengelompokkan mobil bekas ke dalam beberapa segmen "
        "berdasarkan usia, kapasitas mesin, jarak tempuh, tenaga mesin, dan harga."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Mobil", f"{len(df):,}")
    col2.metric("Jumlah Cluster", df["Cluster"].nunique())
    col3.metric("Harga Rata-rata", f"${df['Price'].mean():,.0f}")
    col4.metric("Mileage Rata-rata", f"{df['Mileage'].mean():,.0f} km")

    st.subheader("Distribusi Jumlah Mobil per Cluster")
    cluster_counts = df["Cluster_Label"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Jumlah"]
    fig_bar = px.bar(
        cluster_counts, x="Cluster", y="Jumlah", color="Cluster",
        text="Jumlah", title=None,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Visualisasi Cluster (PCA 2D)")
    fig_pca = px.scatter(
        df, x="PCA1", y="PCA2", color="Cluster_Label",
        hover_data=["Brand", "Model_Year", "Price", "Mileage", "Horsepower"],
        opacity=0.7,
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    with st.expander("Lihat contoh data"):
        st.dataframe(df.head(20), use_container_width=True)

# ==================================================================
# HALAMAN 2: EKSPLORASI CLUSTER
# ==================================================================
elif page == "Eksplorasi Cluster":
    st.title("Eksplorasi & Profil Tiap Cluster")

    selected_cluster = st.selectbox(
        "Pilih Cluster", sorted(df["Cluster"].unique()),
        format_func=lambda c: CLUSTER_LABELS.get(c, f"Cluster {c}"),
    )
    subset = df[df["Cluster"] == selected_cluster]

    st.markdown(f"**Jumlah mobil di cluster ini:** {len(subset):,} "
                f"({len(subset) / len(df) * 100:.1f}% dari total data)")

    profile_cols = ["Car_Age", "Engine_Size", "Mileage", "Horsepower", "Price", "Owner_Count", "Doors"]
    st.subheader("Rata-rata Fitur Numerik")
    st.dataframe(subset[profile_cols].mean().round(2).to_frame("Rata-rata"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Komposisi Brand")
        st.plotly_chart(px.pie(subset, names="Brand", hole=0.4), use_container_width=True)
    with col2:
        st.subheader("Komposisi Fuel Type")
        st.plotly_chart(px.pie(subset, names="Fuel_Type", hole=0.4), use_container_width=True)

    st.subheader("Perbandingan Semua Cluster")
    compare_feat = st.selectbox("Pilih fitur untuk dibandingkan", profile_cols)
    fig_box = px.box(df, x="Cluster_Label", y=compare_feat, color="Cluster_Label")
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Data Mentah pada Cluster Ini")
    st.dataframe(subset, use_container_width=True)

# ==================================================================
# HALAMAN 3: PREDIKSI SATU MOBIL BARU
# ==================================================================
elif page == "Prediksi Mobil Baru":
    st.title("Prediksi Cluster untuk Mobil Baru")
    st.markdown("Masukkan spesifikasi mobil, lalu sistem akan menentukan cluster/segmennya.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            brand = st.selectbox("Brand", sorted(df["Brand"].unique()))
            model_year = st.number_input("Tahun Produksi (Model Year)", min_value=2000, max_value=CURRENT_YEAR, value=2020)
            fuel_type = st.selectbox("Fuel Type", sorted(df["Fuel_Type"].unique()))
        with c2:
            engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
            mileage = st.number_input("Mileage (km)", min_value=0, max_value=400000, value=40000, step=1000)
            transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))
        with c3:
            horsepower = st.number_input("Horsepower", min_value=50, max_value=600, value=200)
            price = st.number_input("Price ($)", min_value=1000, max_value=200000, value=45000, step=500)
            owner_count = st.number_input("Owner Count", min_value=1, max_value=10, value=2)

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

        cluster, pca_coords = predict_cluster(input_row)
        label = CLUSTER_LABELS.get(cluster, f"Cluster {cluster}")

        st.success(f"Mobil ini diprediksi masuk ke **{label}**")

        fig = px.scatter(
            df, x="PCA1", y="PCA2", color="Cluster_Label", opacity=0.4,
        )
        fig.add_scatter(
            x=[pca_coords[0]], y=[pca_coords[1]], mode="markers",
            marker=dict(size=18, color="black", symbol="star"),
            name="Mobil Baru",
        )
        st.plotly_chart(fig, use_container_width=True)

        same_cluster = df[df["Cluster"] == cluster]
        st.markdown(f"**Karakteristik rata-rata mobil pada {label}:**")
        st.dataframe(
            same_cluster[["Car_Age", "Engine_Size", "Mileage", "Horsepower", "Price"]].mean().round(2).to_frame("Rata-rata"),
            use_container_width=True,
        )

# ==================================================================
# HALAMAN 4: PREDIKSI MASSAL DARI FILE CSV
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
            st.error(f"Kolom berikut tidak ditemukan pada file: {missing}")
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
