import streamlit as st
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Regresi Panel Kemiskinan",
    page_icon="📊",
    layout="wide"
)

# --- FUNGSI UNTUK MEMUAT DAN MEMPROSES DATA ---
# Menggunakan cache agar data tidak perlu dimuat ulang setiap kali ada interaksi
@st.cache_data
def load_and_process_data():
    # Membaca data
    df = pd.read_excel("Data_KP.xlsx", sheet_name="dataset")

    # Mengubah tipe data
    df["kode_kabupaten_kota"] = df["kode_kabupaten_kota"].astype(str)
    df["tahun"] = df["tahun"].astype(int)

    # Set MultiIndex
    df.set_index(["kode_kabupaten_kota", "tahun"], inplace=True)

    # Memilih dan mengubah kolom menjadi numerik
    selected_columns = [
        "garis_kemiskinan", "IPM", "TPAK", "upah_minimum", "TPT", "jumlah_penduduk_miskin"
    ]
    df[selected_columns] = df[selected_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=selected_columns, inplace=True)

    # Estimasi model FEM yang terpilih (dari notebook Anda)
    fem_model = PanelOLS.from_formula(
        "garis_kemiskinan ~ IPM + upah_minimum + EntityEffects",
        data=df
    ).fit(cov_type='robust')

    # Ekstrak koefisien dan intersep (fixed effects)
    coefficients = fem_model.params
    intercepts = fem_model.estimated_effects.reset_index()
    intercepts.columns = ['kode_kabupaten_kota', 'tahun', 'intersep']
    
    # Ambil intersep unik per kabupaten/kota (kita asumsikan intersep stabil)
    unique_intercepts = intercepts.groupby('kode_kabupaten_kota')['intersep'].mean().reset_index()

    # Gabungkan nama kabupaten/kota dari data asli
    nama_kabupaten = pd.read_excel("Data_KP.xlsx", sheet_name="dataset")[['kode_kabupaten_kota', 'nama_kabupaten_kota']].drop_duplicates()
    nama_kabupaten['kode_kabupaten_kota'] = nama_kabupaten['kode_kabupaten_kota'].astype(str)
    
    unique_intercepts = pd.merge(unique_intercepts, nama_kabupaten, on='kode_kabupaten_kota')

    return df, fem_model, coefficients, unique_intercepts

# Memuat data menggunakan fungsi yang sudah dibuat
df, fem_model, coefficients, unique_intercepts = load_and_process_data()

# --- TAMPILAN APLIKASI STREAMLIT ---

# Judul Utama
st.title("📊 Aplikasi Analisis Faktor Kemiskinan di Jawa Barat")
st.write("Aplikasi ini dibuat berdasarkan hasil analisis regresi data panel untuk memprediksi **Garis Kemiskinan**.")

# --- SIDEBAR UNTUK INPUT PREDIKSI ---
st.sidebar.header("⚙️ Simulasi Prediksi")

# Mendapatkan daftar nama kabupaten/kota untuk dropdown
list_kabupaten = unique_intercepts['nama_kabupaten_kota'].tolist()
selected_kabupaten_nama = st.sidebar.selectbox("Pilih Kabupaten/Kota:", list_kabupaten)

# Input slider untuk IPM
input_ipm = st.sidebar.slider(
    "Input Nilai Indeks Pembangunan Manusia (IPM):",
    min_value=float(df['IPM'].min()),
    max_value=float(df['IPM'].max()),
    value=float(df['IPM'].mean()),
    step=0.1
)

# Input angka untuk Upah Minimum
input_upah = st.sidebar.number_input(
    "Input Nilai Upah Minimum (Rp):",
    min_value=int(df['upah_minimum'].min()),
    max_value=int(df['upah_minimum'].max() + 1000000), # Beri batas atas lebih
    value=int(df['upah_minimum'].mean()),
    step=50000
)

# --- PERHITUNGAN PREDIKSI ---

# Ambil koefisien dari model
coef_ipm = coefficients['IPM']
coef_upah = coefficients['upah_minimum']

# Ambil intersep untuk kabupaten/kota yang dipilih
selected_intercept_row = unique_intercepts[unique_intercepts['nama_kabupaten_kota'] == selected_kabupaten_nama]
if not selected_intercept_row.empty:
    intersep = selected_intercept_row['intersep'].iloc[0]
else:
    intersep = unique_intercepts['intersep'].mean() # Fallback jika tidak ditemukan


# Hitung prediksi
prediksi = (coef_ipm * input_ipm) + (coef_upah * input_upah) + intersep

# --- TAMPILAN UTAMA (MAIN AREA) ---

# Kolom untuk hasil prediksi dan penjelasan
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Hasil Prediksi")
    st.metric(
        label="Prediksi Garis Kemiskinan (Rp/kapita/bulan)",
        value=f"Rp {prediksi:,.2f}"
    )
    st.info(f"Prediksi ini dihitung untuk **{selected_kabupaten_nama}**.")

with col2:
    st.subheader("Bagaimana Prediksi Dihitung?")
    st.markdown(
        f"""
        Prediksi dihitung menggunakan rumus dari model *Fixed Effect* terpilih:

        **Garis Kemiskinan = (K_IPM × IPM) + (K_Upah × Upah) + Intersep**

        - **K_IPM**: `{coef_ipm:.2f}` (Koefisien IPM)
        - **K_Upah**: `{coef_upah:.4f}` (Koefisien Upah Minimum)
        - **IPM**: `{input_ipm}` (Input Anda)
        - **Upah**: `{input_upah:,.0f}` (Input Anda)
        - **Intersep**: `{intersep:,.2f}` (Nilai unik untuk {selected_kabupaten_nama})
        """
    )


# Expander untuk menampilkan detail model dan data
st.markdown("---")
with st.expander("Lihat Detail Analisis dan Data"):
    st.subheader("Ringkasan Model Fixed Effect (FEM)")
    # Menampilkan ringkasan model dalam format teks
    st.code(fem_model.summary.as_text())
    st.markdown(
        """
        **Interpretasi Penting:**
        - **IPM**: Setiap kenaikan 1 poin IPM, berhubungan dengan kenaikan Garis Kemiskinan sekitar **Rp 16,800**. Ini masuk akal karena IPM yang tinggi seringkali berkorelasi dengan biaya hidup yang lebih tinggi.
        - **Upah Minimum**: Setiap kenaikan Rp 1 pada upah minimum, berhubungan dengan kenaikan Garis Kemiskinan sekitar **Rp 0.12**.
        - **P-value (Prob)** untuk kedua variabel ini sangat kecil (0.0000), yang berarti pengaruhnya **sangat signifikan** secara statistik.
        """
    )

    st.subheader("Statistika Deskriptif Data Asli")
    st.dataframe(df[selected_columns].describe())