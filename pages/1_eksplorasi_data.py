import streamlit as st
from src.data_processing import get_correlation_matrix
from src.visualization import plot_correlation_heatmap, plot_distribution

st.set_page_config(page_title="Eksplorasi Data", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("Eksplorasi Data")

if "df" not in st.session_state or st.session_state.df is None:
    st.warning("Data belum dimuat. Silakan kembali ke halaman utama untuk mengunggah data.")
else:
    df = st.session_state.df
    
    st.subheader("Pratinjau Dataset")
    st.dataframe(df, use_container_width=True, height=250)
    
    tab1, tab2 = st.tabs(["Statistik Deskriptif", "Korelasi Variabel"])
    
    with tab1:
        st.dataframe(df.describe(), use_container_width=True)
        
    with tab2:
        corr = get_correlation_matrix(df)
        fig_corr = plot_correlation_heatmap(corr)
        st.pyplot(fig_corr, clear_figure=True)
        
    st.divider()
    st.subheader("Analisis Distribusi")
    
    col_plot1, col_plot2 = st.columns([1, 2])
    with col_plot1:
        feature_to_plot = st.selectbox("Pilih variabel untuk melihat distribusi:", df.columns)
    with col_plot2:
        fig_dist = plot_distribution(df, feature_to_plot)
        st.pyplot(fig_dist, clear_figure=True)