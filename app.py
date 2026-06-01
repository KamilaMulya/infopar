import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Kepopuleran Destinasi Wisata",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .sub-title    { font-size: 1.1rem; color: #4a6fa5; margin-bottom: 1rem; }
    .metric-card  { background: #f0f4ff; border-radius: 10px; padding: 1rem; text-align: center; }
    .result-box   { border-radius: 12px; padding: 1.5rem; text-align: center; font-size: 1.4rem; font-weight: 700; }
    .result-populer     { background: #d1fae5; color: #065f46; }
    .result-not-populer { background: #fee2e2; color: #991b1b; }
    .stButton > button  { width: 100%; border-radius: 8px; height: 3rem; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────────────────────
BASE = Path("models")

@st.cache_resource(show_spinner="Memuat model...")
def load_artifacts():
    xgb_model = joblib.load(BASE / "xgb_model.pkl")
    rf_model  = joblib.load(BASE / "rf_model.pkl")
    le_cat    = joblib.load(BASE / "label_encoder_category.pkl")
    le_city   = joblib.load(BASE / "label_encoder_city.pkl")
    le_target = joblib.load(BASE / "label_encoder_target.pkl")
    scaler    = joblib.load(BASE / "scaler.pkl")
    with open(BASE / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return xgb_model, rf_model, le_cat, le_city, le_target, scaler, metrics

models_loaded = False
load_error    = ""
try:
    xgb_model, rf_model, le_cat, le_city, le_target, scaler, metrics = load_artifacts()
    models_loaded = True
except Exception as e:
    load_error = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Flag_of_Indonesia.svg/320px-Flag_of_Indonesia.svg.png", width=80)
    st.markdown("## 🏝️ Tourism Classifier")
    st.markdown("Klasifikasi Kepopuleran Destinasi Wisata Indonesia")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "📊 Perbandingan Model", "🔮 Prediksi Kepopuleran"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "**Dataset:** Indonesia Tourism Destination  \n"
        "**Source:** Kaggle · aprabowo  \n"
        "**Model:** XGBoost & Random Forest"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — BERANDA
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Beranda":
    st.markdown('<p class="main-title">🏝️ Klasifikasi Tingkat Kepopuleran Destinasi Wisata</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Perbandingan Algoritma XGBoost dan Random Forest</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">🗺️<br><b>Dataset</b><br>Indonesia Tourism Destination</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🤖<br><b>Algoritma</b><br>XGBoost vs Random Forest</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">🎯<br><b>Task</b><br>Binary Classification</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 Tentang Proyek")
    st.markdown("""
    Aplikasi ini merupakan implementasi dari penelitian **klasifikasi tingkat kepopuleran destinasi wisata**
    di Indonesia menggunakan dua algoritma *machine learning*: **XGBoost** dan **Random Forest**.

    Model dilatih menggunakan dataset **Indonesia Tourism Destination** dari Kaggle yang berisi
    informasi rating pengguna, kategori, lokasi kota, serta harga tiket masuk destinasi wisata.
    Teknik **SMOTE** digunakan untuk menangani ketidakseimbangan kelas, dan **GridSearchCV**
    digunakan untuk menemukan hyperparameter terbaik.
    """)

    st.markdown("### 📋 Fitur yang Digunakan")
    features_df = pd.DataFrame({
        "Fitur"      : ["Category", "City", "Price", "Place_Ratings", "Rating_Count"],
        "Tipe"       : ["Kategorikal", "Kategorikal", "Numerik", "Numerik", "Numerik"],
        "Keterangan" : [
            "Kategori wisata (Alam, Budaya, Taman Hiburan, dll.)",
            "Kota lokasi destinasi wisata",
            "Harga tiket masuk (IDR)",
            "Rating yang diberikan pengguna (skala 1–5)",
            "Jumlah total ulasan/rating yang diterima",
        ]
    })
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.markdown("### 🔖 Definisi Kelas Target")
    c1, c2 = st.columns(2)
    c1.success("✅ **Populer** — Rata-rata rating ≥ 3.0")
    c2.error  ("❌ **Tidak Populer** — Rata-rata rating < 3.0")

    st.markdown("---")
    st.markdown("### 🔄 Alur Pemodelan")
    st.markdown("""
    1. **Preprocessing** → drop missing values, hapus `Time_Minutes`, hitung `Rating_Count` & `Avg_Rating`
    2. **Encoding** → LabelEncoder untuk `Category` dan `City`
    3. **Normalisasi** → StandardScaler untuk fitur numerik
    4. **Balancing** → SMOTE pada data latih
    5. **Training** → GridSearchCV 5-fold CV untuk RF dan XGBoost
    6. **Evaluasi** → Accuracy, Precision, Recall, F1-Score, Confusion Matrix
    """)

    if not models_loaded:
        st.warning(
            "⚠️ **Model belum tersedia.** Jalankan `train_and_save.py` di Google Colab, "
            "lalu upload semua file `.pkl` dan `metrics.json` ke folder `models/`. "
            "Lihat `README.md` untuk panduan lengkap."
        )
    else:
        st.success("✅ Model berhasil dimuat. Silakan gunakan menu di sidebar untuk menjelajahi aplikasi.")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — PERBANDINGAN MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Perbandingan Model":
    st.markdown('<p class="main-title">📊 Perbandingan Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Evaluasi performa XGBoost vs Random Forest pada data uji (20%)</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not models_loaded:
        st.error(f"❌ Model belum tersedia. Error: `{load_error}`")
        st.info("Jalankan `train_and_save.py` di Google Colab lalu upload file hasil training ke folder `models/`.")
        st.stop()

    # ── Metrik tabel ──
    st.subheader("📈 Tabel Metrik Evaluasi")
    metric_keys  = ["accuracy", "precision", "recall", "f1"]
    metric_label = ["Accuracy", "Precision", "Recall", "F1-Score"]

    rf_vals  = [metrics["rf"][k]  for k in metric_keys]
    xgb_vals = [metrics["xgb"][k] for k in metric_keys]

    metrics_df = pd.DataFrame({
        "Metrik"       : metric_label,
        "Random Forest": [f"{v:.4f}" for v in rf_vals],
        "XGBoost"      : [f"{v:.4f}" for v in xgb_vals],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Bar chart ──
    st.subheader("📉 Visualisasi Perbandingan Metrik")
    fig = go.Figure(data=[
        go.Bar(name="Random Forest", x=metric_label, y=rf_vals,  marker_color="#3B82F6",
               text=[f"{v:.4f}" for v in rf_vals],  textposition="outside"),
        go.Bar(name="XGBoost",       x=metric_label, y=xgb_vals, marker_color="#F59E0B",
               text=[f"{v:.4f}" for v in xgb_vals], textposition="outside"),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.12], title="Nilai", tickformat=".2f"),
        xaxis_title="Metrik",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    st.plotly_chart(fig, use_container_width=True)

    # ── Highlight model terbaik ──
    st.markdown("---")
    best_name = "XGBoost" if metrics["xgb"]["accuracy"] >= metrics["rf"]["accuracy"] else "Random Forest"
    best_acc  = max(metrics["xgb"]["accuracy"], metrics["rf"]["accuracy"])
    st.success(f"🏆 **Model Terbaik: {best_name}** dengan akurasi **{best_acc:.4f}** ({best_acc*100:.2f}%)")

    # ── Perbedaan akurasi ──
    diff = abs(metrics["xgb"]["accuracy"] - metrics["rf"]["accuracy"])
    st.info(f"📌 Selisih akurasi antara XGBoost dan Random Forest: **{diff:.4f}** ({diff*100:.2f}%)")

    # ── Confusion Matrix ──
    if "cm_rf" in metrics and "cm_xgb" in metrics:
        st.markdown("---")
        st.subheader("🔢 Confusion Matrix")
        labels = ["Populer", "Tidak Populer"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Random Forest**")
            cm_rf  = np.array(metrics["cm_rf"])
            fig_rf = px.imshow(
                cm_rf, text_auto=True, color_continuous_scale="Blues",
                x=labels, y=labels,
                labels=dict(x="Prediksi", y="Aktual", color="Jumlah")
            )
            fig_rf.update_layout(height=360, margin=dict(t=10))
            st.plotly_chart(fig_rf, use_container_width=True)

        with col2:
            st.markdown("**XGBoost**")
            cm_xgb  = np.array(metrics["cm_xgb"])
            fig_xgb = px.imshow(
                cm_xgb, text_auto=True, color_continuous_scale="Oranges",
                x=labels, y=labels,
                labels=dict(x="Prediksi", y="Aktual", color="Jumlah")
            )
            fig_xgb.update_layout(height=360, margin=dict(t=10))
            st.plotly_chart(fig_xgb, use_container_width=True)

    # ── Kesimpulan ──
    st.markdown("---")
    st.subheader("📝 Kesimpulan Perbandingan")
    st.markdown(f"""
    Berdasarkan hasil evaluasi, **{best_name}** memberikan performa lebih baik secara keseluruhan
    dengan akurasi **{best_acc*100:.2f}%** pada data uji. Oleh karena itu, **{best_name}** dipilih
    sebagai model utama untuk prediksi pada halaman berikutnya.

    Kedua model dilatih dengan teknik **SMOTE** untuk menangani ketidakseimbangan kelas dan
    **GridSearchCV 5-fold** untuk optimasi hyperparameter.
    """)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — PREDIKSI
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Prediksi Kepopuleran":
    st.markdown('<p class="main-title">🔮 Prediksi Kepopuleran Destinasi Wisata</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Masukkan informasi destinasi untuk mendapatkan prediksi kepopuleran</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not models_loaded:
        st.error("❌ Model belum tersedia. Silakan upload file model terlebih dahulu.")
        st.stop()

    categories = sorted(le_cat.classes_.tolist())
    cities     = sorted(le_city.classes_.tolist())

    # ── Input Form ──
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Informasi Destinasi**")
            category = st.selectbox(
                "🏷️ Kategori Destinasi",
                options=categories,
                help="Pilih kategori jenis wisata"
            )
            price = st.number_input(
                "💰 Harga Tiket Masuk (IDR)",
                min_value=0,
                max_value=5_000_000,
                value=50_000,
                step=5_000,
                help="Harga tiket masuk dalam Rupiah"
            )
            place_ratings = st.slider(
                "⭐ Rating Destinasi (1–5)",
                min_value=1.0, max_value=5.0,
                value=4.0, step=0.1,
                help="Rating rata-rata destinasi dari pengunjung"
            )

        with col2:
            st.markdown("**Lokasi & Ulasan**")
            city = st.selectbox(
                "🏙️ Kota",
                options=cities,
                help="Pilih kota lokasi destinasi wisata"
            )
            rating_count = st.number_input(
                "👥 Jumlah Ulasan (Rating_Count)",
                min_value=1,
                max_value=10_000,
                value=100,
                step=10,
                help="Total ulasan/rating yang pernah diberikan pengunjung"
            )
            st.markdown(" ")
            st.markdown(" ")
            model_choice = st.radio(
                "🤖 Pilih Model",
                ["XGBoost (Terbaik)", "Random Forest"],
                help="XGBoost memiliki akurasi lebih tinggi"
            )

        submitted = st.form_submit_button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    # ── Prediction ──
    if submitted:
        # Encode inputs
        try:
            cat_enc = le_cat.transform([category])[0]
        except ValueError:
            cat_enc = 0

        try:
            city_enc = le_city.transform([city])[0]
        except ValueError:
            city_enc = 0

        # Scale numeric features (same order as training: Price, Place_Ratings, Rating_Count)
        num_arr    = np.array([[price, place_ratings, rating_count]], dtype=float)
        num_scaled = scaler.transform(num_arr)

        # Feature vector (same column order as training)
        feature_vec = np.array([[
            num_scaled[0][0],   # Price (scaled)
            num_scaled[0][1],   # Place_Ratings (scaled)
            num_scaled[0][2],   # Rating_Count (scaled)
            cat_enc,            # Category_Encoded
            city_enc            # City_Encoded
        ]])

        # Choose model
        chosen_model = xgb_model if "XGBoost" in model_choice else rf_model

        # Predict
        pred_encoded = chosen_model.predict(feature_vec)[0]
        proba        = chosen_model.predict_proba(feature_vec)[0]
        label        = le_target.inverse_transform([pred_encoded])[0]
        confidence   = float(np.max(proba))
        prob_populer = float(proba[le_target.transform(["Populer"])[0]])

        # ── Display result ──
        st.markdown("---")
        st.subheader("📋 Hasil Prediksi")

        res_col, conf_col = st.columns([2, 1])

        with res_col:
            if label == "Populer":
                st.markdown(
                    f'<div class="result-box result-populer">✅ {label}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-box result-not-populer">❌ {label}</div>',
                    unsafe_allow_html=True
                )

        with conf_col:
            st.metric("Confidence", f"{confidence:.2%}")
            st.metric("Prob. Populer", f"{prob_populer:.2%}")

        # Progress bar
        st.markdown("##### Probabilitas Kepopuleran")
        st.progress(prob_populer, text=f"Populer: {prob_populer:.2%}")

        # Input summary
        st.markdown("---")
        st.markdown("##### 📌 Ringkasan Input")
        summary_df = pd.DataFrame({
            "Parameter"  : ["Kategori", "Kota", "Harga Tiket", "Rating", "Jumlah Ulasan", "Model"],
            "Nilai"      : [category, city, f"Rp {price:,.0f}", f"{place_ratings:.1f} / 5.0",
                            str(rating_count), model_choice]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Contoh penggunaan ──
    with st.expander("💡 Contoh Destinasi untuk Dicoba"):
        st.markdown("""
        | Kategori        | Kota        | Harga (IDR) | Rating | Ulasan |
        |----------------|-------------|-------------|--------|--------|
        | Taman Hiburan  | Jakarta     | 150,000     | 4.5    | 500    |
        | Budaya         | Yogyakarta  | 25,000      | 4.2    | 300    |
        | Alam           | Bandung     | 10,000      | 3.8    | 200    |
        | Bahari         | Surabaya    | 5,000       | 2.5    | 50     |
        """)
