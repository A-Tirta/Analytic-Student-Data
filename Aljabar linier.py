import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup Tema Aplikasi #
st.set_page_config(
    page_title="Analisis Efisiensi Pembelajaran Hibrida",
    page_icon="📊",
    layout="wide"
)

# Gaya CSS Custom #
st.markdown("""
    <style>
    body {
        background-color: #121212;
    }
    .css-18e3th9 {
        padding: 1rem;
    }
    .stApp {
        background-color: #121212;
        border-radius: 10px;
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f3f4f6;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Header Aplikasi #
st.markdown("<h1 style='text-align: center; color: #E0E0E0;'>📊 Analisis Efisiensi Pembelajaran Hibrida</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #E0E0E0;'>Model Aljabar Linier dalam Sistem Perkuliahan</h4>", unsafe_allow_html=True)
st.write("---")

# Upload Data CSV #
st.sidebar.header("📂 Upload Data CSV")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

# Jika tidak ada file, gunakan data contoh #
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Menggunakan data contoh karena tidak ada file yang diunggah.")
    data = {
        "Jam Tatap Muka": [10, 12, 8, 15, 11, 14, 9, 13, 7, 16, 10, 12, 8, 14, 11, 13, 9, 15, 7, 16, 10, 12, 8, 15, 11, 14, 9],
        "Jam Online": [15, 14, 18, 12, 16, 10, 20, 13, 19, 11, 17, 15, 18, 10, 16, 14, 20, 12, 19, 11, 14, 13, 17, 10, 18, 12, 15],
        "Nilai Akhir": [80, 85, 75, 90, 82, 88, 70, 87, 68, 92, 78, 85, 74, 89, 81, 86, 72, 91, 67, 93, 79, 84, 76, 89, 83, 87, 71]
    }
    df = pd.DataFrame(data)

# Tampilkan Data #
st.subheader("📌 Data Mahasiswa")
st.dataframe(df.style.set_properties(**{'background-color': '#f9f9f9', 'color': 'black'}))

# Statistik Deskriptif #
st.subheader("📊 Statistik Deskriptif")
st.write(df.describe().T.style.set_properties(**{'background-color': '#ffffff', 'color': 'black'}))

# Heatmap Korelasi #
st.subheader("📈 Korelasi Antar Variabel")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Matriks Korelasi", fontsize=12, color="black")
st.pyplot(fig)

# Distribusi Data #
st.subheader("📊 Distribusi Nilai Mahasiswa")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(df["Nilai Akhir"], kde=True, bins=10, color="blue", alpha=0.7)
ax.set_xlabel("Nilai Akhir", color="black", fontsize=12)
ax.set_ylabel("Frekuensi", color="black", fontsize=12)
st.pyplot(fig)

# Pilih Variabel untuk Analisis #
st.sidebar.subheader("⚙️ Pilih Variabel")
x_vars = st.sidebar.multiselect("Pilih variabel independen:", ["Jam Tatap Muka", "Jam Online"], default=["Jam Tatap Muka"])
y_var = "Nilai Akhir"

if not x_vars:
    st.warning("Harap pilih minimal satu variabel independen untuk regresi!")
else:
    X = df[x_vars]
    y = df[y_var]

    # Split Data untuk Pelatihan dan Pengujian #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih Model #
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Hasil Analisis #
    st.subheader("📊 Hasil Analisis Regresi Linier")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-Squared", f"{r2_score(y_test, y_pred):.2f}")
    col2.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
    col3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    col4.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    # Persamaan Regresi #
    coef_str = " + ".join([f"{model.coef_[i]:.2f} × {x_vars[i]}" for i in range(len(x_vars))])
    st.write(f"<p style='color: #E0E0E0;'><strong>Persamaan Regresi:</strong> Y = {coef_str} + {model.intercept_:.2f}</p>", unsafe_allow_html=True)

    # Grafik Prediksi vs Aktual #
    st.subheader("📉 Prediksi vs Aktual")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, label="Prediksi vs Aktual", color="red", alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "--", color="black")  # Garis ideal
    ax.set_xlabel("Nilai Aktual", color="black", fontsize=12)
    ax.set_ylabel("Nilai Prediksi", color="black", fontsize=12)
    ax.legend()
    st.pyplot(fig)

    # Boxplot Analisis Variabel #
    st.subheader("📦 Distribusi Variabel Terhadap Nilai Akhir")
    selected_feature = st.selectbox("Pilih variabel untuk analisis:", x_vars)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df[selected_feature], y=df[y_var], palette="coolwarm", ax=ax)
    ax.set_xlabel(selected_feature, color="black", fontsize=12)
    ax.set_ylabel(y_var, color="black", fontsize=12)
    st.pyplot(fig)

    # Prediksi Interaktif #
    st.subheader("🧮 Prediksi Nilai Akhir")
    input_values = []
    for x in x_vars:
        val = st.number_input(f"Masukkan {x}:", min_value=0, max_value=50, value=10)
        input_values.append(val)

    if st.button("Prediksi"):
        prediksi_nilai = model.predict([input_values])[0]
        st.success(f"📢 Prediksi Nilai Akhir: **{prediksi_nilai:.2f}**")
