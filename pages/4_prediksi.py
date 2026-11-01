import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Prediksi", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .prediction-card { background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 2rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("Prediksi Nilai Akhir")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Data belum dimuat. Silakan kembali ke halaman utama.")
else:
    df = st.session_state.df
    
    if df.shape[0] < 5:
        st.error("Data tidak mencukupi untuk memprediksi.")
    else:
        X = df[["Jam Tatap Muka", "Jam Online"]]
        y = df["Nilai Akhir"]
        
        model = LinearRegression()
        model.fit(X, y)
        
        st.markdown("Masukkan estimasi jam pembelajaran untuk melihat proyeksi nilai akhir mahasiswa.")
        st.write("")
        
        min_tm, max_tm = float(X["Jam Tatap Muka"].min()), float(X["Jam Tatap Muka"].max())
        min_ol, max_ol = float(X["Jam Online"].min()), float(X["Jam Online"].max())
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                input_tm = st.number_input("Jam Tatap Muka", value=min_tm, help=f"Rentang data latih: {min_tm} - {max_tm}")
            with col2:
                input_ol = st.number_input("Jam Online", value=min_ol, help=f"Rentang data latih: {min_ol} - {max_ol}")
            
            submit_button = st.form_submit_button("Hitung Prediksi", type="primary")
            
        if submit_button:
            is_extrapolation = (input_tm < min_tm or input_tm > max_tm or input_ol < min_ol or input_ol > max_ol)
            
            if is_extrapolation:
                st.warning("Perhatian: Nilai berada di luar rentang observasi. Hasil prediksi dapat berisiko kurang akurat.")
                
            prediction = model.predict(pd.DataFrame({"Jam Tatap Muka": [input_tm], "Jam Online": [input_ol]}))[0]
            
            st.write("")
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="margin-bottom:0; color:#166534;">Estimasi Nilai Akhir</h3>
                <h1 style="font-size: 3rem; margin-top:0; color:#15803d;">{prediction:.2f}</h1>
            </div>
            """, unsafe_allow_html=True)