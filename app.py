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
    # Kita gunakan semua variabel signifikan dan tidak signifikan untuk ringkasan,
    # tapi hanya yang signifikan untuk prediksi.
    fem_model = PanelOLS.from_formula(
        "garis_kemiskinan ~ IPM + TPAK + upah_minimum + TPT + jumlah_penduduk_miskin + EntityEffects",
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

    # Kembalikan semua variabel yang dibutuhkan di luar fungsi
    return df, fem_model, coefficients, unique_intercepts, selected_columns # <-- DIUBAH

# Memuat data menggunakan fungsi yang sudah dibuat
# Menangkap variabel selected_columns yang sudah dikembalikan
df, fem_model, coefficients, unique_intercepts, selected_columns = load_and_process_data() # <-- DIUBAH

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
st.sidebar.markdown("**Input Variabel Signifikan:**")
input_ipm = st.sidebar.slider(
    "Indeks Pembangunan Manusia (IPM):",
    min_value=float(df['IPM'].min()),
    max_value=float(df['IPM'].max()),
    value=float(df['IPM'].mean()),
    step=0.1
)

# Upah Minimum
input_upah = st.sidebar.slider(
    "Geser untuk memilih Upah Minimum (Rp):",
    min_value=1000000,  # Batas bawah 1 juta
    max_value=6000000,  # Batas atas 6 juta
    value=2500000,      # Nilai awal yang ditampilkan
    step=500000,        # Slider akan melompat per 500 ribu
    format="Rp %,.0f"   # Format angka agar mudah dibaca 
)

# Ambil koefisien dari model HANYA UNTUK VARIABEL SIGNIFIKAN
coef_ipm = coefficients['IPM']
coef_upah = coefficients['upah_minimum']

# Ambil intersep untuk kabupaten/kota yang dipilih
selected_intercept_row = unique_intercepts[unique_intercepts['nama_kabupaten_kota'] == selected_kabupaten_nama]
intersep = selected_intercept_row['intersep'].iloc[0] if not selected_intercept_row.empty else unique_intercepts['intersep'].mean()

# Hitung prediksi HANYA berdasarkan variabel signifikan
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
        """
        Prediksi dihitung menggunakan rumus dari model *Fixed Effect* dengan variabel yang signifikan:

        **Garis Kemiskinan = (`K_IPM` × `IPM`) + (`K_Upah` × `Upah`) + `Intersep`**
        """
    )
    st.code(
        f"Garis Kemiskinan = ({coef_ipm:.2f} × {input_ipm}) + ({coef_upah:.4f} × {input_upah:,.0f}) + ({intersep:,.2f})",
        language='bash'
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
        - **IPM**: Setiap kenaikan 1 poin IPM, berhubungan dengan kenaikan Garis Kemiskinan.
        - **Upah Minimum**: Setiap kenaikan Rp 1 pada upah minimum, berhubungan dengan kenaikan Garis Kemiskinan.
        - **P-value (Prob)** untuk `IPM` dan `upah_minimum` sangat kecil (0.0000), yang berarti pengaruhnya **sangat signifikan** secara statistik dan layak digunakan untuk prediksi.
        """
    )

    st.subheader("Statistika Deskriptif Data Asli")
    # Baris ini sekarang tidak akan error lagi
    st.dataframe(df[selected_columns].describe())
