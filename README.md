# 🏝️ Klasifikasi Tingkat Kepopuleran Destinasi Wisata Indonesia
### Perbandingan Algoritma XGBoost dan Random Forest

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📖 Deskripsi

Aplikasi web berbasis **Streamlit** untuk mengklasifikasikan tingkat kepopuleran destinasi wisata di Indonesia.
Proyek ini membandingkan dua algoritma *machine learning*: **XGBoost** dan **Random Forest**, menggunakan
dataset [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination) dari Kaggle.

---

## 📁 Struktur Folder

```
├── app.py                 
├── requirements.txt        
├── README.md
├── .gitignore
└── models/                 
    ├── xgb_model.pkl
    ├── rf_model.pkl
    ├── label_encoder_category.pkl
    ├── label_encoder_city.pkl
    ├── label_encoder_target.pkl
    ├── scaler.pkl
    └── metrics.json
```

---

## 🚀 Cara Deploy ke Streamlit (Step by Step)

### STEP 1 — Training Model di Google Colab

1. Buka **Google Colab** dan upload file `train_and_save.py`
2. Jalankan semua cell:
   ```python
   # Di Colab, jalankan:
   !python train_and_save.py
   ```
3. Setelah selesai, file-file berikut akan otomatis ter-download:
   - `xgb_model.pkl`
   - `rf_model.pkl`
   - `label_encoder_category.pkl`
   - `label_encoder_city.pkl`
   - `label_encoder_target.pkl`
   - `scaler.pkl`
   - `metrics.json`

> ⚠️ **Pastikan Kaggle API sudah di-setup di Colab:**
> ```python
> from google.colab import files
> files.upload()  # Upload kaggle.json dari https://www.kaggle.com/settings
> !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
> ```

---

### STEP 2 — Buat Repository GitHub

1. Buka [github.com](https://github.com) → **New Repository**
2. Beri nama repo (contoh: `tourism-popularity-classifier`)
3. Pilih **Public**, centang **Add README**
4. Klik **Create Repository**

---

### STEP 3 — Upload File ke GitHub

1. Setelah repo dibuat, klik **Add file → Upload files**
2. Upload semua file berikut ke **root** repo:
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
3. Buat folder `models/` dengan cara klik **Add file → Create new file**, ketik `models/.gitkeep`, lalu commit
4. Masuk ke folder `models/`, klik **Add file → Upload files**, upload semua file `.pkl` dan `metrics.json`

> 💡 **Atau gunakan Git via terminal:**
> ```bash
> git clone https://github.com/username/tourism-popularity-classifier.git
> cd tourism-popularity-classifier
> # Salin semua file ke sini
> mkdir models
> # Salin file .pkl dan metrics.json ke folder models/
> git add .
> git commit -m "Initial commit: add Streamlit app and model files"
> git push origin main
> ```

---

### STEP 4 — Deploy ke Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub
3. Klik **New app**
4. Isi form:
   - **Repository**: `username/tourism-popularity-classifier`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Klik **Deploy!**
6. Tunggu beberapa menit hingga app berjalan ✅

---

## 🖥️ Fitur Aplikasi

| Halaman | Deskripsi |
|---|---|
| 🏠 Beranda | Informasi proyek, fitur, dan alur pemodelan |
| 📊 Perbandingan Model | Tabel metrik, bar chart, dan confusion matrix kedua model |
| 🔮 Prediksi | Form interaktif untuk prediksi kepopuleran destinasi wisata |

---

## 🧪 Metodologi

| Tahap | Keterangan |
|---|---|
| Dataset | Indonesia Tourism Destination (Kaggle) |
| Target | Populer (avg rating ≥ 3.0) / Tidak Populer |
| Fitur | Category, City, Price, Place_Ratings, Rating_Count |
| Encoding | LabelEncoder untuk fitur kategorikal |
| Normalisasi | StandardScaler untuk fitur numerik |
| Balancing | SMOTE pada data latih |
| Tuning | GridSearchCV 5-fold cross validation |
| Evaluasi | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

---

## 📦 Dependensi

```
streamlit, numpy, pandas, scikit-learn, xgboost, joblib, imbalanced-learn, plotly
```

Install lokal (opsional):
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 👤 Author

**Kamila Mulya Fadila** — Informatika
