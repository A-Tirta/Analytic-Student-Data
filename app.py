import streamlit as st
import pandas as pd
from src.data_processing import validate_schema, clean_data

st.set_page_config(
    page_title="Analisis Efisiensi Pembelajaran",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("Analisis Efisiensi Pembelajaran Hibrida")
st.markdown("Aplikasi ini menerapkan pemodelan regresi linier dan aljabar linier untuk menganalisis hubungan antara jam pembelajaran dengan nilai akhir mahasiswa secara komprehensif.")
st.divider()

st.sidebar.header("Konfigurasi Data")
uploaded_file = st.sidebar.file_uploader("Unggah file dataset CSV", type=["csv"])

def load_default_data():
    return pd.read_csv("data/sample_data.csv")

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        is_valid, missing_cols = validate_schema(raw_df)
        
        if not is_valid:
            st.error(f"Dataset tidak valid. Kolom yang hilang: {', '.join(missing_cols)}")
            st.session_state.df = None
        else:
            cleaned_df, stats = clean_data(raw_df)
            st.session_state.df = cleaned_df
            st.sidebar.success("Data berhasil diunggah dan dibersihkan.")
            
            if stats["missing_removed"] > 0 or stats["duplicates_removed"] > 0:
                st.sidebar.info(
                    f"Pembersihan otomatis:\n"
                    f"- Baris kosong dihapus: {stats['missing_removed']}\n"
                    f"- Duplikat dihapus: {stats['duplicates_removed']}"
                )
    except Exception as e:
        st.error("Terjadi kesalahan saat membaca file CSV. Pastikan format file benar.")
        st.session_state.df = None
else:
    st.sidebar.info("Menggunakan data sampel bawaan sistem.")
    st.session_state.df = load_default_data()

if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Ikhtisar Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Observasi", df.shape[0])
    col2.metric("Rata-rata Nilai Akhir", f"{df['Nilai Akhir'].mean():.2f}")
    col3.metric("Nilai Maksimum", df['Nilai Akhir'].max())
    col4.metric("Nilai Minimum", df['Nilai Akhir'].min())
    
    st.write("")
    with st.expander("Panduan Navigasi Aplikasi", expanded=True):
        st.markdown("""
        Gunakan menu di bilah sisi untuk melakukan eksplorasi lebih lanjut:
        - **Eksplorasi Data**: Analisis distribusi dan korelasi antar variabel.
        - **Model Regresi**: Evaluasi kinerja model regresi linier secara statistik.
        - **Aljabar Linier**: Memahami representasi matriks dan perhitungan algoritma.
        - **Prediksi**: Memasukkan data baru untuk memprediksi probabilitas nilai akhir.
        
        *Catatan: Jika dataset yang digunakan berskala kecil, hasil analisis bersifat eksploratori.*
        """)
else:
    st.warning("Silakan unggah dataset yang valid untuk memulai analisis.")