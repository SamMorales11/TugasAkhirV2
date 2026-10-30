# Estimasi Nilai Pasar Pemain Sepak Bola Profesional
### Implementasi Random Forest Regression + Genetic Algorithm

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

Proyek Tugas Akhir untuk mengestimasi nilai pasar (market value) pemain sepak bola profesional di lima liga top Eropa berdasarkan statistik kinerja musiman menggunakan **Random Forest Regression** yang dioptimasi dengan **Algoritma Genetika (GASearchCV)**.

---

## Hasil Utama

| Model                    | R²     | RMSE              |
|--------------------------|--------|-------------------|
| Random Forest Default    | 0.4583 | € 13.797.607,78  |
| Random Forest + GA       | **0.4824** | **€ 13.487.799,54** |

- Paired T-test: **p-value = 0.5312** (tidak signifikan secara statistik)
- Fitur paling berpengaruh: **Age**, **SCA**, **Progressive Passes (PrgP)**

---

## Struktur Project
```bash
TugasAkhirV2/
├── data/
│   ├── raw/                  # Data mentah (FBref + Transfermarkt)
│   └── processed/            # Data siap model
├── models/                   # Model yang sudah dilatih (.joblib)
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   └── 2_pemodelan.ipynb
├── results/
│   └── figures/              # Semua visualisasi
├── predict.py                # Script prediksi standalone
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Preprocessing
Buka dan jalankan semua cell di:
```bash
notebooks/1_data_preprocessing.ipynb
```

### 3 Jalankan Modeling
Buka dan jalankan semua cell di:
```bash
notebooks/2_pemodelan.ipynb
```

### 4 Prediksi Pemain Baru (Opsional)
```bash
python predict.py --age 24 --sca 90 --prgp 150 --xg 7.2 --pos MF --liga "Premier League"
```