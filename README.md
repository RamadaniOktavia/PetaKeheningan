# 🗺️ Peta Keheningan Global

Analisis Spasio-Temporal Keterwakilan Perempuan di Parlemen Nasional dan Proyeksi Paritas Gender Menggunakan Model ARIMA (1997–2025)

Proyek Akhir — Mata Kuliah Visualisasi Data Spasio-Temporal
Program Studi Informatika, Fakultas Teknologi Informasi, Universitas Andalas

**Oktavia Ramadani** · NIM 2311533002

---

## Deskripsi Proyek

**Peta Keheningan Global** adalah aplikasi visualisasi data spasio-temporal interaktif yang merepresentasikan dinamika keterwakilan perempuan di parlemen nasional dari **185 negara** selama periode **1997–2025**. Aplikasi ini sekaligus mengintegrasikan model **ARIMA per negara** untuk memproyeksikan tahun pencapaian paritas gender (50% kursi perempuan) di tiap negara.

Aplikasi mengintegrasikan empat sumber data dengan peran berbeda:

| Sumber | Peran |
|---|---|
| [Our World in Data (OWID)](https://ourworldindata.org/women-in-parliaments) | Deret waktu utama keterwakilan perempuan di parlemen |
| [World Bank — SG.GEN.PARL.ZS](https://data.worldbank.org/indicator/SG.GEN.PARL.ZS) | Validasi silang nilai persentase kursi perempuan |
| [International IDEA — Gender Quotas Database](https://www.idea.int/data-tools/data/gender-quotas) | Diferensiasi jenis kuota gender per negara |
| [Natural Earth GeoJSON](https://github.com/datasets/geo-countries) | Batas administrasi spasial negara |

**Fitur utama:**
- Peta choropleth interaktif dengan slider tahun (1997–2025), gradien putih → ungu
- Mode proyeksi ARIMA: peta kategoris (putih / hijau / oranye / merah) berdasarkan perkiraan tahun paritas
- Panel detail negara: persentase kursi, jenis kuota, RMSE/MAE model, grafik tren mini
- Grafik tren regional multi-line (5 kawasan: Amerika, Asia-Pasifik, Afrika, Eropa, MENA)
- Marker momen historis (tahun pertama melewati ambang 30% dan 50%)
- Insight tambahan: korelasi jenis kuota gender vs kecepatan pertumbuhan

**Tech stack:** Python · Streamlit · Folium · Plotly · Pandas · Shapely · pmdarima

Artikel ilmiah lengkap yang membahas latar belakang, tinjauan pustaka, metodologi, dan hasil penelitian tersedia di repositori ini / pada berkas terlampir tugas akhir.

---

## Struktur Folder

```
PetaKeheningan/
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Daftar dependensi Python
├── data/
│   ├── panel_data_clean.csv          # Data panel: iso3, country, region, year, pct_women, quota_type
│   ├── map_data_with_forecast.csv    # Hasil model ARIMA: iso3, country, region, quota_type,
│   │                                  # pct_2025, rmse, mae, model_order, predicted_parity_year, category, note
│   └── countries.geojson             # Batas administrasi negara (Natural Earth)
├── Notebook/
│   ├── Preprocessing_PetaKeheningan.ipynb   # Akuisisi, pembersihan, validasi silang, & integrasi data
│   └── PipelineML_PetaKeheningan.ipynb      # Pipeline ARIMA per negara: training, evaluasi, forecasting
├── .streamlit/
│   └── config.toml         # Konfigurasi tampilan Streamlit
├── .gitignore
└── README.md
```

> **Catatan kolom GeoJSON:** `app.py` mendeteksi otomatis nama properti kode ISO3 dan nama negara, karena penamaan ini berbeda-beda antar rilis Natural Earth (`ISO_A3`, `ISO3166-1-Alpha-3`, dll). Tidak perlu penyesuaian manual.

### Notebook

| Notebook | Isi |
|---|---|
| `Notebook/Preprocessing_PetaKeheningan.ipynb` | Akuisisi data dari OWID, World Bank, IDEA, dan Natural Earth; pembersihan nilai hilang, validasi silang OWID vs World Bank, integrasi quota_type, normalisasi kode ISO3, hingga menghasilkan `panel_data_clean.csv`. |
| `Notebook/PipelineML_PetaKeheningan.ipynb` | Pipeline ARIMA per negara menggunakan `auto_arima` (pmdarima): seleksi order (p, d, q), walk-forward validation (RMSE/MAE), forecasting hingga tahun paritas, fallback regresi linear, dan ekspor `map_data_with_forecast.csv`. |

Kedua notebook dapat dijalankan secara berurutan di Google Colab atau Jupyter lokal untuk mereproduksi seluruh pipeline data dari sumber mentah hingga berkas yang dikonsumsi `app.py`.

---

## Instalasi

1. **Clone repositori**

   ```bash
   git clone https://github.com/RamadaniOktavia/PetaKeheningan.git
   cd PetaKeheningan
   ```

2. **(Opsional) Buat virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependensi**

   ```bash
   pip install -r requirements.txt
   ```

---

## Menjalankan Aplikasi

Pastikan ketiga berkas data (`panel_data_clean.csv`, `map_data_with_forecast.csv`, `countries.geojson`) sudah ada di dalam folder `data/`, lalu jalankan:

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

### Deploy ke Streamlit Cloud (opsional)

1. Push repositori ini ke akun GitHub kamu (jika belum)
2. Buka [share.streamlit.io](https://share.streamlit.io), login dengan akun GitHub yang sama
3. Klik **New app** → pilih repositori `PetaKeheningan` → branch `main` → file utama `app.py`
4. Klik **Deploy** — link permanen akan tersedia setelah build selesai (±2–5 menit)

---

## Mereproduksi Pipeline Data (Notebook)

Berkas di `data/` sudah merupakan hasil akhir pipeline dan siap dipakai langsung oleh `app.py`. Untuk mereproduksi prosesnya dari data mentah:

1. Buka `Notebook/Preprocessing_PetaKeheningan.ipynb` di Google Colab atau Jupyter, jalankan seluruh sel untuk menghasilkan `panel_data_clean.csv`.
2. Lanjutkan dengan `Notebook/PipelineML_PetaKeheningan.ipynb`, yang membaca `panel_data_clean.csv` dan menghasilkan `map_data_with_forecast.csv` (hasil forecasting ARIMA per negara beserta metrik RMSE/MAE).
3. Salin kedua berkas CSV hasil keluaran ke folder `data/` sebelum menjalankan `streamlit run app.py`.

---

## Metodologi Singkat

Setiap negara dimodelkan secara independen menggunakan **ARIMA** (`auto_arima` dari pustaka `pmdarima`), dengan fallback regresi linear sederhana untuk negara dengan data tidak memadai (<10 titik data valid). Evaluasi model menggunakan skema *walk-forward validation* (latih 1997–2020, uji 2021–2025), dilaporkan melalui metrik **RMSE** dan **MAE** dalam satuan poin persentase. Proyeksi paritas dikategorikan ke dalam empat kelas:

- 🟢 **Hijau** — paritas sebelum 2063
- 🟠 **Oranye** — paritas 2063–2100
- 🔴 **Merah** — paritas setelah 2100 / tidak konvergen
- ⚪ **Putih** — sudah mencapai ≥ 50%

Penjelasan lengkap pipeline pra-pemrosesan, pemodelan, dan evaluasi tersedia pada artikel ilmiah pendamping proyek ini.

---

## Lisensi & Atribusi Data

Kode sumber proyek ini disusun untuk keperluan akademik (Tugas Akhir mata kuliah Visualisasi Data Spasio-Temporal). Seluruh data eksternal yang digunakan tetap berada di bawah lisensi masing-masing penyedia (Our World in Data, World Bank Open Data, International IDEA, Natural Earth).
