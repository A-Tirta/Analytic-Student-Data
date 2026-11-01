import streamlit as st
import numpy as np
from src.linear_algebra import compute_design_matrix, compute_coefficients_pinv, format_matrix_latex

st.set_page_config(page_title="Aljabar Linier", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .math-box { background-color: #ffffff; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("Aljabar Linier")
st.markdown("Model regresi linier berganda diekspresikan menggunakan notasi matriks untuk komputasi yang efisien.")

with st.container():
    st.latex(r"y = X\beta + \epsilon")
    st.markdown("""
    - **y**: Vektor target
    - **X**: Matriks Desain
    - **β**: Vektor koefisien
    - **ε**: Vektor residu
    """)

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Unggah data terlebih dahulu di halaman utama untuk melihat kalkulasi matriks.")
else:
    df = st.session_state.df
    X_df = df[["Jam Tatap Muka", "Jam Online"]]
    y_series = df["Nilai Akhir"]
    
    X_design = compute_design_matrix(X_df)
    beta = compute_coefficients_pinv(X_design, y_series)
    
    st.divider()
    st.subheader("Representasi Matriks Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Matriks Desain X**")
        st.latex(format_matrix_latex(X_design, max_rows=5, name="X"))
    
    with col2:
        st.markdown("**Vektor Target y**")
        st.latex(format_matrix_latex(y_series.to_numpy().reshape(-1, 1), max_rows=5, name="y"))
        
    st.divider()
    st.subheader("Penyelesaian Least Squares")
    
    col_theory, col_result = st.columns(2)
    with col_theory:
        st.markdown("Penyelesaian dengan **Invers Semu Moore-Penrose** untuk stabilitas komputasi:")
        st.latex(r"\beta = X^+ y")
    with col_result:
        st.markdown("**Hasil Perhitungan Koefisien:**")
        st.latex(format_matrix_latex(beta.reshape(-1, 1), max_rows=3, name=r"\beta"))