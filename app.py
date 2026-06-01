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
    page_title="Wisata Classifier — Kepopuleran Destinasi",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🗺</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Icons (inline SVG strings) ────────────────────────────────────────────────
ICONS = {
    "home":    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>',
    "bar":     '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>',
    "output":  '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>',
    "map":     '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"/><path stroke-linecap="round" stroke-linejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"/></svg>',
    "star":    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
    "tag":     '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A2 2 0 013 12V7a4 4 0 014-4z"/></svg>',
    "ticket":  '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M15 5v2m0 4v2m0 4v2M5 5a2 2 0 00-2 2v3a2 2 0 110 4v3a2 2 0 002 2h14a2 2 0 002-2v-3a2 2 0 110-4V7a2 2 0 00-2-2H5z"/></svg>',
    "users":   '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"/></svg>',
    "cpu":     '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "check":   '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="#059669" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>',
    "x":       '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="#dc2626" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>',
    "logo":    '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="none" viewBox="0 0 24 24" stroke="#2563EB" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>',
    "filter":  '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"/></svg>',
    "list":    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 10h16M4 14h16M4 18h16"/></svg>',
}

def icon(name, extra_style=""):
    svg = ICONS.get(name, "")
    return f'<span style="display:inline-flex;align-items:center;vertical-align:middle;{extra_style}">{svg}</span>'

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0f172a 0%, #1e3a5f 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 0.55rem 0.9rem;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: background 0.18s;
    font-weight: 500;
    font-size: 0.92rem;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08); }

/* Page background */
.main .block-container { padding-top: 2rem; max-width: 1200px; }

/* Page title */
.page-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 0.3rem;
}
.page-title {
    font-size: 1.85rem; font-weight: 800;
    color: #0f172a; letter-spacing: -0.03em;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.page-subtitle {
    font-size: 0.95rem; color: #64748b;
    margin-bottom: 1.5rem; margin-left: 2px;
}

/* Cards */
.info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    height: 100%;
}
.info-card .card-icon {
    width: 42px; height: 42px;
    border-radius: 10px;
    background: #eff6ff;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.8rem;
}
.info-card .card-label {
    font-size: 0.78rem; font-weight: 600;
    color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.info-card .card-value {
    font-size: 1.05rem; font-weight: 700; color: #0f172a;
}

/* Result box */
.result-box {
    border-radius: 14px; padding: 1.6rem;
    text-align: center; font-size: 1.5rem; font-weight: 800;
    letter-spacing: -0.02em;
    display: flex; align-items: center; justify-content: center; gap: 10px;
}
.result-populer     { background: #ecfdf5; color: #065f46; border: 2px solid #6ee7b7; }
.result-not-populer { background: #fef2f2; color: #991b1b; border: 2px solid #fca5a5; }

/* Destination card */
.dest-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    display: flex; align-items: flex-start; gap: 12px;
    transition: box-shadow 0.18s;
}
.dest-card:hover { box-shadow: 0 4px 16px rgba(37,99,235,0.10); }
.dest-rank {
    min-width: 32px; height: 32px; border-radius: 8px;
    background: #eff6ff; color: #2563EB;
    font-weight: 700; font-size: 0.85rem;
    display: flex; align-items: center; justify-content: center;
}
.dest-name  { font-weight: 700; font-size: 1rem; color: #0f172a; margin-bottom: 2px; }
.dest-meta  { font-size: 0.83rem; color: #64748b; display: flex; flex-wrap: wrap; gap: 10px; margin-top: 4px; }
.dest-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: #f1f5f9; border-radius: 6px;
    padding: 2px 8px; font-size: 0.78rem; font-weight: 600; color: #475569;
}

/* Section header */
.section-header {
    font-size: 1.05rem; font-weight: 700; color: #0f172a;
    margin: 1.4rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 7px;
}
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.4rem 0; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #f8fafc;
    border-radius: 10px; padding: 4px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px; padding: 0.45rem 1.1rem;
    font-weight: 600; font-size: 0.9rem;
    color: #64748b !important;
}
.stTabs [aria-selected="true"] {
    background: #2563EB !important;
    color: white !important;
}

/* Streamlit overrides */
.stButton > button {
    border-radius: 9px; height: 2.9rem; font-size: 0.95rem;
    font-weight: 600; width: 100%;
}
.stButton > button[kind="primary"] {
    background: #2563EB; border: none;
}
.stButton > button[kind="primary"]:hover { background: #1d4ed8; }
.stSelectbox label, .stNumberInput label, .stSlider label, .stRadio label {
    font-weight: 600; font-size: 0.88rem; color: #374151;
}
div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 800; color: #0f172a; }
div[data-testid="stMetricDelta"] { font-size: 0.82rem; }
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

@st.cache_data(show_spinner=False)
def load_dataset():
    """Auto-detect tourism destination CSV — pilih file dengan kolom paling lengkap."""
    # Kolom penting destinasi (makin banyak cocok = makin prioritas)
    PRIORITY_COLS = {
        "place_name", "placename",
        "category", "kategori",
        "price", "harga",
        "place_ratings", "place_rating",
        "rating_count", "jumlah_ulasan",
        "lat", "latitude",
        "long", "lon", "longitude",
        "description", "deskripsi",
    }
    # Kolom wajib minimal ada — kalau tidak ada salah satunya, skip file ini
    REQUIRED_COLS = {"place_name", "placename", "category", "kategori", "price", "harga"}
    # File yang diketahui BUKAN dataset destinasi — skip
    SKIP_FILENAMES = {"package_tourism", "tourism_rating", "user", "users"}

    search_dirs = [Path("data"), Path(".")]
    candidates = []
    for d in search_dirs:
        if d.exists():
            candidates += sorted(d.glob("*.csv"))

    best_df    = None
    best_score = -1

    for p in candidates:
        # Skip file yang diketahui bukan dataset destinasi
        stem_lower = p.stem.lower().replace("-", "_").replace(" ", "_")
        if any(skip in stem_lower for skip in SKIP_FILENAMES):
            continue
        try:
            df_peek   = pd.read_csv(p, nrows=5)
            cols_lower = {c.strip().lower() for c in df_peek.columns}
            # Harus punya minimal satu kolom wajib
            if not (cols_lower & REQUIRED_COLS):
                continue
            # Skor = jumlah kolom prioritas yang cocok
            score = len(cols_lower & PRIORITY_COLS)
            if score > best_score:
                best_score = score
                best_df    = pd.read_csv(p)
        except Exception:
            continue
    return best_df

models_loaded = False
load_error    = ""
try:
    xgb_model, rf_model, le_cat, le_city, le_target, scaler, metrics = load_artifacts()
    models_loaded = True
except Exception as e:
    load_error = str(e)

dataset = load_dataset()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:0.6rem 0 1rem 0;">
        {ICONS['logo']}
        <div>
            <div style="font-size:1rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.02em;">Wisata Classifier</div>
            <div style="font-size:0.73rem;color:#94a3b8;font-weight:400;">Indonesia Tourism</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.08);margin-bottom:1rem;"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["Beranda", "Metriks Model", "Output"],
        label_visibility="collapsed",
        format_func=lambda x: {
            "Beranda":       f"  Beranda",
            "Metriks Model": f"  Metriks Model",
            "Output":        f"  Output",
        }[x]
    )

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — BERANDA
# ═════════════════════════════════════════════════════════════════════════════
if page == "Beranda":
    st.markdown(f"""
    <div class="page-header">
        {ICONS['logo']}
        <span class="page-title">Klasifikasi Kepopuleran Destinasi Wisata</span>
    </div>
    <div class="page-subtitle">Perbandingan Algoritma XGBoost dan Random Forest untuk prediksi destinasi wisata Indonesia</div>
    """, unsafe_allow_html=True)

    # Stat cards
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("map",      "Dataset",    "Indonesia Tourism Destination", "#eff6ff", "#2563EB"),
        ("cpu",      "Algoritma",  "XGBoost & Random Forest",        "#f0fdf4", "#16a34a"),
        ("tag",      "Task",       "Binary Classification",           "#fef9c3", "#ca8a04"),
        ("filter",   "Balancing",  "SMOTE + GridSearchCV 5-fold",     "#fdf4ff", "#9333ea"),
    ]
    for col, (ic, lbl, val, bg, clr) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="info-card">
                <div class="card-icon" style="background:{bg};">
                    <span style="color:{clr};">{ICONS[ic]}</span>
                </div>
                <div class="card-label">{lbl}</div>
                <div class="card-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        st.markdown(f'<div class="section-header">{icon("list")} Tentang Proyek</div>', unsafe_allow_html=True)
        st.markdown("""
        Aplikasi ini merupakan implementasi penelitian **klasifikasi tingkat kepopuleran destinasi wisata** di Indonesia
        menggunakan dua algoritma *machine learning*: **XGBoost** dan **Random Forest**.

        Model dilatih menggunakan dataset **Indonesia Tourism Destination** dari Kaggle yang berisi informasi
        rating pengguna, kategori, lokasi kota, serta harga tiket masuk. Teknik **SMOTE** digunakan untuk
        menangani ketidakseimbangan kelas, dan **GridSearchCV** untuk menemukan hyperparameter terbaik.
        """)

        st.markdown(f'<div class="section-header">{icon("list")} Alur Pemodelan</div>', unsafe_allow_html=True)
        steps = [
            ("Preprocessing", "Drop missing values, hapus Time_Minutes, hitung Rating_Count & Avg_Rating"),
            ("Encoding",       "LabelEncoder untuk Category dan City"),
            ("Normalisasi",    "StandardScaler untuk fitur numerik"),
            ("Balancing",      "SMOTE pada data latih untuk menyeimbangkan kelas"),
            ("Training",       "GridSearchCV 5-fold CV untuk RF dan XGBoost"),
            ("Evaluasi",       "Accuracy, Precision, Recall, F1-Score, Confusion Matrix"),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:10px;align-items:flex-start;">
                <div style="min-width:26px;height:26px;border-radius:50%;background:#2563EB;
                    color:white;font-size:0.75rem;font-weight:700;display:flex;
                    align-items:center;justify-content:center;">{i}</div>
                <div>
                    <div style="font-weight:700;font-size:0.9rem;color:#0f172a;">{title}</div>
                    <div style="font-size:0.83rem;color:#64748b;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<div class="section-header">{icon("tag")} Fitur yang Digunakan</div>', unsafe_allow_html=True)
        features_df = pd.DataFrame({
            "Fitur"      : ["Category", "City", "Price", "Place_Ratings", "Rating_Count"],
            "Tipe"       : ["Kategorikal", "Kategorikal", "Numerik", "Numerik", "Numerik"],
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True, height=220)

        st.markdown(f'<div class="section-header">{icon("star")} Definisi Kelas Target</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#ecfdf5;border:1.5px solid #6ee7b7;border-radius:10px;padding:0.9rem 1rem;margin-bottom:0.7rem;">
            <div style="font-weight:700;color:#065f46;margin-bottom:2px;">Populer</div>
            <div style="font-size:0.85rem;color:#047857;">Rata-rata rating pengguna &ge; 3.0</div>
        </div>
        <div style="background:#fef2f2;border:1.5px solid #fca5a5;border-radius:10px;padding:0.9rem 1rem;">
            <div style="font-weight:700;color:#991b1b;margin-bottom:2px;">Tidak Populer</div>
            <div style="font-size:0.85rem;color:#b91c1c;">Rata-rata rating pengguna &lt; 3.0</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    if not models_loaded:
        st.warning(
            f"**Model belum tersedia.** Jalankan `train_and_save.py` di Google Colab, "
            "lalu upload semua file `.pkl` dan `metrics.json` ke folder `models/`. "
            "Lihat `README.md` untuk panduan lengkap."
        )
    else:
        st.success("Model berhasil dimuat. Gunakan menu di sidebar untuk menjelajahi fitur klasifikasi dan prediksi.")

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — METRIKS MODEL
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Metriks Model":
    st.markdown(f"""
    <div class="page-header">
        {ICONS['bar']}
        <span class="page-title">Metriks Model</span>
    </div>
    <div class="page-subtitle">Evaluasi performa XGBoost vs Random Forest pada data uji (20%)</div>
    """, unsafe_allow_html=True)

    if not models_loaded:
        st.error(f"Model belum tersedia. Error: `{load_error}`")
        st.info("Jalankan `train_and_save.py` di Google Colab lalu upload file hasil training ke folder `models/`.")
        st.stop()

    metric_keys  = ["accuracy", "precision", "recall", "f1"]
    metric_label = ["Accuracy", "Precision", "Recall", "F1-Score"]
    rf_vals      = [metrics["rf"][k]  for k in metric_keys]
    xgb_vals     = [metrics["xgb"][k] for k in metric_keys]

    # Headline metrics
    best_name = "XGBoost" if metrics["xgb"]["accuracy"] >= metrics["rf"]["accuracy"] else "Random Forest"
    best_acc  = max(metrics["xgb"]["accuracy"], metrics["rf"]["accuracy"])
    diff      = abs(metrics["xgb"]["accuracy"] - metrics["rf"]["accuracy"])

    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val in zip(
        [m1, m2, m3, m4],
        ["Akurasi XGBoost", "Akurasi Random Forest", "Model Terbaik", "Selisih Akurasi"],
        [f"{metrics['xgb']['accuracy']:.2%}", f"{metrics['rf']['accuracy']:.2%}", best_name, f"{diff:.4f}"]
    ):
        col.metric(lbl, val)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Bar chart
    st.markdown(f'<div class="section-header">{icon("bar")} Perbandingan Metrik Evaluasi</div>', unsafe_allow_html=True)
    fig = go.Figure(data=[
        go.Bar(name="Random Forest", x=metric_label, y=rf_vals,  marker_color="#3B82F6",
               text=[f"{v:.4f}" for v in rf_vals],  textposition="outside"),
        go.Bar(name="XGBoost",       x=metric_label, y=xgb_vals, marker_color="#F59E0B",
               text=[f"{v:.4f}" for v in xgb_vals], textposition="outside"),
    ])
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.15], title="Nilai", tickformat=".2f"),
        xaxis_title="Metrik",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig, use_container_width=True)

    # Tabel metrik
    st.markdown(f'<div class="section-header">{icon("list")} Tabel Metrik Lengkap</div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        "Metrik"       : metric_label,
        "Random Forest": [f"{v:.4f}" for v in rf_vals],
        "XGBoost"      : [f"{v:.4f}" for v in xgb_vals],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Confusion Matrix
    if "cm_rf" in metrics and "cm_xgb" in metrics:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">{icon("bar")} Confusion Matrix</div>', unsafe_allow_html=True)
        labels = ["Populer", "Tidak Populer"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Random Forest**")
            cm_rf  = np.array(metrics["cm_rf"])
            fig_rf = px.imshow(cm_rf, text_auto=True, color_continuous_scale="Blues",
                               x=labels, y=labels,
                               labels=dict(x="Prediksi", y="Aktual", color="Jumlah"))
            fig_rf.update_layout(height=340, margin=dict(t=10), font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig_rf, use_container_width=True)
        with col2:
            st.markdown("**XGBoost**")
            cm_xgb  = np.array(metrics["cm_xgb"])
            fig_xgb = px.imshow(cm_xgb, text_auto=True, color_continuous_scale="Oranges",
                                x=labels, y=labels,
                                labels=dict(x="Prediksi", y="Aktual", color="Jumlah"))
            fig_xgb.update_layout(height=340, margin=dict(t=10), font=dict(family="Plus Jakarta Sans"))
            st.plotly_chart(fig_xgb, use_container_width=True)

    # Kesimpulan
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">{icon("list")} Kesimpulan</div>', unsafe_allow_html=True)
    st.markdown(f"""
    Berdasarkan hasil evaluasi, **{best_name}** memberikan performa lebih baik secara keseluruhan
    dengan akurasi **{best_acc:.2%}** pada data uji. Kedua model dilatih dengan teknik **SMOTE**
    untuk menangani ketidakseimbangan kelas dan **GridSearchCV 5-fold** untuk optimasi hyperparameter.
    Selisih akurasi antara keduanya adalah **{diff:.4f}** ({diff*100:.2f}%).
    """)

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — OUTPUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Output":
    st.markdown(f"""
    <div class="page-header">
        {ICONS['output']}
        <span class="page-title">Output</span>
    </div>
    <div class="page-subtitle">Klasifikasi destinasi wisata populer & prediksi kepopuleran destinasi baru</div>
    """, unsafe_allow_html=True)

    if not models_loaded:
        st.error("Model belum tersedia. Silakan upload file model terlebih dahulu.")
        st.stop()

    tab1, tab2 = st.tabs(["  Klasifikasi Destinasi", "  Prediksi Kepopuleran"])

    # ──────────────────────────────────────────────────────────────────────────
    #  TAB 1 — KLASIFIKASI (filter & tampilkan destinasi dari dataset)
    # ──────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f'<div class="section-header">{icon("filter")} Filter Destinasi</div>', unsafe_allow_html=True)

        if dataset is None:
            st.warning(
                "Dataset tidak ditemukan. Pastikan minimal satu file `.csv` ada di folder `data/` "
                "dengan kolom seperti `Place_Name`, `Category`, `City`, `Price`, `Place_Ratings`, `Rating_Count`."
            )
        else:
            # Normalise column names
            df = dataset.copy()
            df.columns = [c.strip() for c in df.columns]

            # Flexible column detection - exact match dulu, lalu partial/substring
            def find_col(df, *variants):
                # Pass 1: exact match (case-insensitive, strip whitespace)
                for c in df.columns:
                    if c.strip().lower() in variants:
                        return c
                # Pass 2: substring match
                for c in df.columns:
                    cl = c.strip().lower().replace(" ", "_")
                    for v in variants:
                        if v in cl or cl in v:
                            return c
                return None

            name_col   = find_col(df,
                "place_name", "placename", "place name",
                "nama_wisata", "nama_tempat", "nama", "name", "tempat")
            cat_col    = find_col(df,
                "category", "kategori", "cat", "type", "tipe", "jenis")
            city_col   = find_col(df,
                "city", "kota", "kabupaten", "daerah", "location", "lokasi")
            price_col  = find_col(df,
                "price", "harga", "ticket_price", "price_idr", "tiket",
                "entrance_fee", "harga_tiket", "biaya")
            rating_col = find_col(df,
                "place_ratings", "place_rating", "rating", "avg_rating",
                "average_rating", "ratings", "nilai", "score")
            cnt_col    = find_col(df,
                "rating_count", "jumlah_ulasan", "count", "num_ratings",
                "total_ratings", "ulasan", "review_count", "reviews",
                "jumlah_rating")
            lat_col    = find_col(df,
                "lat", "latitude", "lintang")
            lon_col    = find_col(df,
                "long", "lon", "lng", "longitude", "bujur")
            desc_col   = find_col(df,
                "description", "deskripsi", "desc", "keterangan", "detail")

            # Filter inputs
            fc1, fc2 = st.columns(2)
            with fc1:
                sel_cat = st.multiselect(
                    "Jenis Destinasi",
                    options=sorted(df[cat_col].dropna().unique().tolist()) if cat_col else [],
                    placeholder="Semua kategori",
                )
                if price_col:
                    price_min_v = int(df[price_col].min())
                    price_max_v = int(df[price_col].max())
                    step_p = max(1000, (price_max_v - price_min_v) // 100)
                    price_range = st.slider(
                        "Rentang Harga Tiket Masuk (IDR)",
                        min_value=price_min_v, max_value=price_max_v,
                        value=(price_min_v, price_max_v), step=step_p,
                    )
                else:
                    price_range = None
                    st.caption("Kolom harga tidak ditemukan di dataset.")

            with fc2:
                sel_city = st.multiselect(
                    "Daerah / Kota",
                    options=sorted(df[city_col].dropna().unique().tolist()) if city_col else [],
                    placeholder="Semua kota",
                )
                if cnt_col:
                    min_cnt = st.number_input(
                        "Minimal Jumlah Ulasan",
                        min_value=int(df[cnt_col].min()),
                        max_value=int(df[cnt_col].max()),
                        value=int(df[cnt_col].min()), step=10,
                    )
                else:
                    min_cnt = None
                    st.caption("Kolom jumlah ulasan tidak ditemukan di dataset.")

            if rating_col:
                min_rating = st.slider(
                    "Minimal Rating Destinasi",
                    min_value=float(round(df[rating_col].min(), 1)),
                    max_value=float(round(df[rating_col].max(), 1)),
                    value=float(round(df[rating_col].min(), 1)),
                    step=0.1,
                )
            else:
                min_rating = None
                st.caption("Kolom rating tidak ditemukan di dataset.")

            run_btn = st.button("Tampilkan Hasil Klasifikasi", type="primary")

            if not run_btn:
                st.markdown(
                    '<div style="color:#64748b;font-size:0.88rem;margin-top:0.4rem;">' +
                    'Atur filter di atas lalu klik tombol untuk melihat hasil klasifikasi.</div>',
                    unsafe_allow_html=True
                )
            else:
                # Apply filters
                filtered = df.copy()
                if cat_col  and sel_cat:   filtered = filtered[filtered[cat_col].isin(sel_cat)]
                if city_col and sel_city:  filtered = filtered[filtered[city_col].isin(sel_city)]
                if price_col and price_range:
                    filtered = filtered[(filtered[price_col] >= price_range[0]) &
                                        (filtered[price_col] <= price_range[1])]
                if cnt_col and min_cnt is not None:
                    filtered = filtered[filtered[cnt_col] >= min_cnt]
                if rating_col and min_rating is not None:
                    filtered = filtered[filtered[rating_col] >= min_rating]

                best_model = xgb_model if metrics["xgb"]["accuracy"] >= metrics["rf"]["accuracy"] else rf_model

                if len(filtered) == 0:
                    st.info("Tidak ada destinasi yang sesuai filter. Coba perluas kriteria pencarian.")
                else:
                    # Build feature matrix for the filtered rows
                    preds = []
                    for _, row in filtered.iterrows():
                        try:
                            cat_enc  = le_cat.transform([row[cat_col]])[0]  if cat_col  else 0
                            city_enc = le_city.transform([row[city_col]])[0] if city_col else 0
                            pr       = float(row[price_col])  if price_col  else 0
                            rt       = float(row[rating_col]) if rating_col else 0
                            rc       = float(row[cnt_col])    if cnt_col    else 0
                            num_arr  = np.array([[pr, rt, rc]], dtype=float)
                            scaled   = scaler.transform(num_arr)[0]
                            fvec     = np.array([[scaled[0], scaled[1], scaled[2], cat_enc, city_enc]])
                            pred_enc = best_model.predict(fvec)[0]
                            label    = le_target.inverse_transform([pred_enc])[0]
                            preds.append(label)
                        except Exception:
                            preds.append("Unknown")
    
                    filtered = filtered.copy()
                    filtered["_Prediksi"] = preds
    
                    populer_df    = filtered[filtered["_Prediksi"] == "Populer"].reset_index(drop=True)
                    tdk_populer_df = filtered[filtered["_Prediksi"] == "Tidak Populer"].reset_index(drop=True)
    
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
                    # Summary chips
                    st.markdown(f"""
                    <div style="display:flex;gap:12px;margin-bottom:1.2rem;flex-wrap:wrap;">
                        <div style="background:#ecfdf5;border:1.5px solid #6ee7b7;border-radius:10px;
                            padding:0.6rem 1.1rem;display:flex;align-items:center;gap:8px;">
                            {ICONS['check']}
                            <span style="font-weight:700;color:#065f46;">Populer: {len(populer_df)} destinasi</span>
                        </div>
                        <div style="background:#fef2f2;border:1.5px solid #fca5a5;border-radius:10px;
                            padding:0.6rem 1.1rem;display:flex;align-items:center;gap:8px;">
                            {ICONS['x']}
                            <span style="font-weight:700;color:#991b1b;">Tidak Populer: {len(tdk_populer_df)} destinasi</span>
                        </div>
                        <div style="background:#f1f5f9;border:1.5px solid #cbd5e1;border-radius:10px;
                            padding:0.6rem 1.1rem;">
                            <span style="font-weight:600;color:#475569;">Total hasil filter: {len(filtered)} destinasi</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
                    def render_dest_map(dest_df, label, accent, bg, border, icon_html):
                        """Render list + map for a group of destinations."""
                        if len(dest_df) == 0:
                            st.info(f"Tidak ada destinasi {label} dengan filter ini.")
                            return
    
                        col_list, col_map_view = st.columns([1, 1], gap="large")
    
                        with col_list:
                            st.markdown(f"""
                            <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.8rem;">
                                {icon_html}
                                <span style="font-weight:700;font-size:0.95rem;color:#0f172a;">
                                    Daftar Destinasi {label}
                                </span>
                                <span style="background:{bg};border:1px solid {border};border-radius:6px;
                                    padding:2px 10px;font-size:0.8rem;font-weight:700;color:{accent};">
                                    {len(dest_df)}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
    
                            for i, (_, row) in enumerate(dest_df.head(30).iterrows(), start=1):
                                nm  = row[name_col]  if name_col  else f"Destinasi {i}"
                                ct  = row[city_col]  if city_col  else "-"
                                cat = row[cat_col]   if cat_col   else "-"
                                pr  = f"Rp {int(row[price_col]):,}" if price_col else "-"
                                rt  = f"{row[rating_col]:.1f}" if rating_col else "-"
                                rc  = f"{int(row[cnt_col]):,} ulasan" if cnt_col else ""
                                card_border = f"border-left:3px solid {accent};"
                                st.markdown(f"""
                                <div class="dest-card" style="{card_border}">
                                    <div class="dest-rank" style="background:{bg};color:{accent};">{i}</div>
                                    <div style="flex:1">
                                        <div class="dest-name">{nm}</div>
                                        <div class="dest-meta">
                                            <span class="dest-badge">{ICONS['map']} {ct}</span>
                                            <span class="dest-badge">{ICONS['tag']} {cat}</span>
                                            <span class="dest-badge">{ICONS['ticket']} {pr}</span>
                                            <span class="dest-badge">{ICONS['star']} {rt} &nbsp;·&nbsp; {rc}</span>
                                        </div>
                                        {f'<div style="font-size:0.8rem;color:#64748b;margin-top:6px;">{str(row[desc_col])[:120]}...</div>' if desc_col and pd.notna(row[desc_col]) else ""}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
    
                        with col_map_view:
                            st.markdown(f'<div class="section-header">{icon("map")} Peta Lokasi — {label}</div>', unsafe_allow_html=True)
                            if lat_col and lon_col and name_col:
                                map_cols = [c for c in [name_col, lat_col, lon_col, city_col, rating_col, cat_col] if c]
                                map_df   = dest_df[map_cols].dropna(subset=[lat_col, lon_col]).head(30).copy()
                                rename   = {name_col: "Nama", lat_col: "lat", lon_col: "lon"}
                                if city_col:   rename[city_col]   = "Kota"
                                if rating_col: rename[rating_col] = "Rating"
                                if cat_col:    rename[cat_col]    = "Kategori"
                                map_df.rename(columns=rename, inplace=True)
    
                                if len(map_df) > 0:
                                    color_scale = "Greens" if label == "Populer" else "Reds"
                                    fig_map = px.scatter_mapbox(
                                        map_df,
                                        lat="lat", lon="lon",
                                        hover_name="Nama" if "Nama" in map_df.columns else None,
                                        hover_data={c: True for c in ["Kota","Rating","Kategori"] if c in map_df.columns},
                                        color="Rating" if "Rating" in map_df.columns else None,
                                        color_continuous_scale=color_scale,
                                        zoom=4.5,
                                        height=480,
                                    )
                                    fig_map.update_traces(marker=dict(size=12))
                                    fig_map.update_layout(
                                        mapbox_style="open-street-map",
                                        margin=dict(l=0, r=0, t=0, b=0),
                                        coloraxis_colorbar=dict(title="Rating"),
                                        font=dict(family="Plus Jakarta Sans"),
                                    )
                                    st.plotly_chart(fig_map, use_container_width=True)
                            else:
                                st.info("Kolom latitude/longitude tidak ditemukan di dataset.")
    
                    # Render Populer
                    st.markdown(f'<div class="section-header" style="color:#065f46;">{ICONS["check"]} Destinasi Populer</div>', unsafe_allow_html=True)
                    render_dest_map(populer_df, "Populer", "#059669", "#ecfdf5", "#6ee7b7", ICONS["check"])
    
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
                    # Render Tidak Populer
                    st.markdown(f'<div class="section-header" style="color:#991b1b;">{ICONS["x"]} Destinasi Tidak Populer</div>', unsafe_allow_html=True)
                    render_dest_map(tdk_populer_df, "Tidak Populer", "#dc2626", "#fef2f2", "#fca5a5", ICONS["x"])

    # ──────────────────────────────────────────────────────────────────────────
    #  TAB 2 — PREDIKSI
    # ──────────────────────────────────────────────────────────────────────────
    with tab2:
        categories = sorted(le_cat.classes_.tolist())
        cities     = sorted(le_city.classes_.tolist())

        st.markdown(f'<div class="section-header">{icon("filter")} Informasi Destinasi</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            category = st.selectbox("Kategori Destinasi", options=categories,
                                    help="Pilih kategori jenis wisata")
            price = st.number_input("Harga Tiket Masuk (IDR)", min_value=0, max_value=5_000_000,
                                    value=50_000, step=5_000)
            place_ratings = st.slider("Rating Destinasi (1–5)", min_value=1.0, max_value=5.0,
                                      value=4.0, step=0.1)

        with col2:
            city = st.selectbox("Kota / Daerah", options=cities)
            rating_count = st.number_input("Jumlah Ulasan", min_value=1, max_value=10_000,
                                           value=100, step=10)
            model_choice = st.radio("Pilih Model", ["XGBoost (Terbaik)", "Random Forest"],
                                    help="XGBoost umumnya memiliki akurasi lebih tinggi")

        predict_btn = st.button("Jalankan Prediksi", type="primary", use_container_width=False)

        if predict_btn:
            try:
                cat_enc  = le_cat.transform([category])[0]
            except ValueError:
                cat_enc  = 0
            try:
                city_enc = le_city.transform([city])[0]
            except ValueError:
                city_enc = 0

            num_arr    = np.array([[price, place_ratings, rating_count]], dtype=float)
            num_scaled = scaler.transform(num_arr)
            feature_vec = np.array([[
                num_scaled[0][0],
                num_scaled[0][1],
                num_scaled[0][2],
                cat_enc,
                city_enc,
            ]])

            chosen_model = xgb_model if "XGBoost" in model_choice else rf_model
            pred_encoded = chosen_model.predict(feature_vec)[0]
            proba        = chosen_model.predict_proba(feature_vec)[0]
            label        = le_target.inverse_transform([pred_encoded])[0]
            confidence   = float(np.max(proba))
            prob_populer = float(proba[le_target.transform(["Populer"])[0]])

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-header">{icon("output")} Hasil Prediksi</div>', unsafe_allow_html=True)

            res_col, stat_col = st.columns([2, 1], gap="large")
            with res_col:
                icon_html = ICONS["check"] if label == "Populer" else ICONS["x"]
                css_cls   = "result-populer" if label == "Populer" else "result-not-populer"
                st.markdown(
                    f'<div class="result-box {css_cls}">{icon_html}&nbsp;{label}</div>',
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Probabilitas Kepopuleran**")
                st.progress(prob_populer, text=f"Populer: {prob_populer:.2%}")
                st.progress(1 - prob_populer, text=f"Tidak Populer: {1-prob_populer:.2%}")

            with stat_col:
                st.metric("Kelas Prediksi",    label)
                st.metric("Confidence",        f"{confidence:.2%}")
                st.metric("Prob. Populer",     f"{prob_populer:.2%}")
                st.metric("Prob. Tidak Populer", f"{1-prob_populer:.2%}")

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-header">{icon("list")} Ringkasan Input</div>', unsafe_allow_html=True)
            summary_df = pd.DataFrame({
                "Parameter": ["Kategori", "Kota", "Harga Tiket", "Rating", "Jumlah Ulasan", "Model"],
                "Nilai":     [category, city, f"Rp {price:,.0f}", f"{place_ratings:.1f} / 5.0",
                              str(rating_count), model_choice],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with st.expander("Contoh Destinasi untuk Dicoba"):
            ex_df = pd.DataFrame({
                "Kategori":    ["Taman Hiburan", "Budaya",     "Alam",    "Bahari"],
                "Kota":        ["Jakarta",        "Yogyakarta", "Bandung", "Surabaya"],
                "Harga (IDR)": [150_000,          25_000,       10_000,    5_000],
                "Rating":      [4.5,               4.2,          3.8,       2.5],
                "Ulasan":      [500,               300,          200,       50],
            })
            st.dataframe(ex_df, use_container_width=True, hide_index=True)