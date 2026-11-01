import streamlit as st
from src.modeling import train_and_evaluate
from src.visualization import plot_actual_vs_predicted

st.set_page_config(page_title="Model Regresi", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Hasil Analisis Regresi Linier")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Data belum dimuat. Silakan kembali ke halaman utama.")
else:
    df = st.session_state.df
    
    if df.shape[0] < 5:
        st.error("Jumlah data terlalu sedikit untuk melatih model regresi. Diperlukan minimal 5 baris data.")
    else:
        model, metrics, X_test, y_test, y_pred = train_and_evaluate(df)
        
        st.subheader("Metrik Evaluasi Model")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R-Squared", f"{metrics['r2']:.4f}")
        col2.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
        col3.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
        col4.metric("Mean Squared Error", f"{metrics['mse']:.2f}")
        
        st.divider()
        
        col_eq, col_viz = st.columns([1, 1.2])
        
        with col_eq:
            st.subheader("Persamaan Regresi")
            coef_tm = model.coef_[0]
            coef_ol = model.coef_[1]
            intercept = model.intercept_
            
            st.code(f"Nilai Akhir = {intercept:.2f} + ({coef_tm:.2f} × Jam Tatap Muka) + ({coef_ol:.2f} × Jam Online)", language="text")
            
            with st.expander("Lihat Interpretasi Koefisien", expanded=True):
                st.markdown(f"""
                - **Intersep {intercept:.2f}**: Estimasi nilai dasar apabila seluruh jam pembelajaran adalah nol.
                - **Jam Tatap Muka {coef_tm:.2f}**: Peningkatan nilai per tambahan 1 jam tatap muka.
                - **Jam Online {coef_ol:.2f}**: Peningkatan nilai per tambahan 1 jam online.
                """)
                
        with col_viz:
            st.subheader("Visualisasi Prediksi")
            fig = plot_actual_vs_predicted(y_test, y_pred)
            st.pyplot(fig, clear_figure=True)