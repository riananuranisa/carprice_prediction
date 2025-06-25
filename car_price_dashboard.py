import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans

# -----------------------------
# SET PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
with open('model_rf.pkl', 'rb') as f:
    model_rf = pickle.load(f)

with open('model_kmeans.pkl', 'rb') as f:
    model_kmeans = pickle.load(f)

with open('scaler_kmeans.pkl', 'rb') as f:
    scaler_kmeans = pickle.load(f)

with open('cluster_labels.json', 'r') as f:
    cluster_label_map = json.load(f)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("CarPrice_Assignment.csv")

    if 'CarName' in df.columns:
        df[['CarBrand', 'CarModel']] = df['CarName'].str.split(' ', n=1, expand=True)
    else:
        df['CarBrand'] = 'unknown'
        df['CarModel'] = 'unknown'

    brand_corrections = {
        'toyouta': 'toyota',
        'Nissan': 'nissan',
        'maxda': 'mazda',
        'vw': 'volkswagen',
        'vokswagen': 'volkswagen',
        'porcshce': 'porsche',
    }
    df['CarBrand'] = df['CarBrand'].replace(brand_corrections).str.lower()

    df_clean = df.copy()
    df_clean[['CarBrand', 'CarModel']] = df_clean['CarName'].str.split(' ', n=1, expand=True)
    df_clean['CarBrand'] = df_clean['CarBrand'].str.strip().str.lower()
    df_clean.drop(columns=['car_ID', 'CarName', 'CarModel'], errors='ignore', inplace=True)
    
    return df, df_clean

df_rekom, df = load_data()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
    <div style="background-color:#ff4b4b;padding:10px;border-radius:5px">
        <h2 style="color:white;text-align:center">Sistem Prediksi Harga Mobil</h2>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
menu = st.sidebar.radio("📂 Navigasi Menu", [
    "Home",
    "Business Understanding",
    "Data Understanding",
    "Data Exploration",
    "Evaluasi Model",
    "Prediksi Harga Mobil"
])

# -----------------------------
# MENU: HOME
# -----------------------------
if menu == "Home":
    st.subheader("📋 Home")
    st.markdown("""
    Dashboard ini menyajikan proses **data mining** untuk memprediksi harga mobil berbasis spesifikasi teknis kendaraan. Disusun Oleh:
    - (220102034) Fauzan Aditia Putra
    - (220102052) Muhamad Wendi Narizal P.
    - (220102062) Muhammad Rifqi Firdaus     
    - (220102072) Riana Nur Anisa            

    Komponen utama:
    - 📘 Business Understanding
    - 📊 Data Understanding & Exploration
    - 🤖 Modeling & Evaluasi
    - 🔍 Prediksi harga berdasarkan input pengguna

    Dataset yang digunakan adalah [**CarPrice_Assignment.csv**](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction), yang diunduh dari Kaggle.

    """)

# -----------------------------
# MENU: BUSINESS UNDERSTANDING
# -----------------------------
elif menu == "Business Understanding":
    st.subheader("📘 Business Understanding")

    from PIL import Image
    image = Image.open("image.png")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=400, use_container_width=False)

    st.markdown("""
    Proyek ini menangani kebutuhan prediksi harga mobil secara objektif dan efisien berbasis data historis.

    **🔍 Masalah Utama:**  
    Sulitnya menentukan harga mobil secara akurat, terutama bagi:
    - Dealer mobil bekas  
    - Konsumen pembeli mobil  
    - Platform otomotif online  

    **🎯 Tujuan Proyek:**  
    - Prediksi harga mobil dengan regresi (Random Forest)  
    - Segmentasi mobil dengan clustering (K-Means)  
    - Evaluasi model dengan MAE, RMSE, R² & Silhouette Score  
    - Visualisasi interaktif dengan Streamlit  

    **💡 Manfaat:**  
    - Estimasi harga lebih objektif  
    - Analisis segmen pasar  
    - Antarmuka pengguna yang interaktif  
    """)

# -----------------------------
# MENU: DATA UNDERSTANDING
# -----------------------------
elif menu == "Data Understanding":
    st.subheader("📊 Data Understanding")

    st.markdown("""
    Dataset yang digunakan adalah [**CarPrice_Assignment.csv**](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction), yang berisi 205 data mobil dari berbagai merek 
    beserta spesifikasi teknis dan informasi umum lainnya.  
    Setiap baris mewakili satu unit mobil, tanpa missing value, sehingga siap untuk dianalisis setelah proses 
    encoding dan scaling.

    **Karakteristik Dataset:**
    - Jumlah baris: 205  
    - Jumlah kolom: 26 kolom asli + 2 kolom hasil ekstraksi (`CarBrand`, `CarModel`)  
    - Tipe data:  
        - Numerik: dimensi kendaraan, performa, efisiensi, dan harga  
        - Kategorikal: jenis bahan bakar, tipe bodi, penggerak, dan lainnya  
        - Unik dan simbolik: `car_ID` (ID mobil), `symboling` (skor risiko asuransi)  
    """)

    st.write("Jumlah baris:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])

    st.subheader("Tipe Data")
    st.code(df.dtypes.astype(str))

    st.subheader("Cek Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Distribusi Brand Mobil")
    st.dataframe(df['CarBrand'].value_counts())

    st.markdown("""
    Dataset ini digunakan sebagai dasar dalam membangun model prediksi harga serta segmentasi mobil berdasarkan kemiripan spesifikasi.
    """)

# -----------------------------
# MENU: DATA EXPLORATION
# -----------------------------
elif menu == "Data Exploration":
    st.subheader("🔎 Data Exploration")

    # 1. Distribusi Harga Mobil
    st.markdown("### Distribusi Harga Mobil")
    st.markdown("""
    Distribusi harga mobil dalam dataset bersifat **right-skewed**, artinya sebagian besar mobil memiliki harga lebih rendah, 
    dengan beberapa mobil premium di rentang harga tinggi.  
    Pola ini mencerminkan kondisi nyata pasar mobil dan cocok untuk model prediksi.
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['price'], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_title("Distribusi Harga Mobil")
    ax.set_xlabel("Harga")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

    # 2. Korelasi Fitur Numerik
    st.markdown("### Korelasi Fitur Numerik")
    st.markdown("""
    Analisis korelasi menunjukkan bahwa fitur seperti `curbweight`, `enginesize`, dan `horsepower` memiliki korelasi tertinggi terhadap harga.  
    Sebaliknya, fitur efisiensi bahan bakar (`citympg`, `highwaympg`) menunjukkan korelasi negatif terhadap harga.
    """)
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Heatmap Korelasi")
    st.pyplot(fig)

    # 3. Scatterplot Hubungan Fitur Numerik vs Harga
    st.markdown("### Hubungan Harga dengan Fitur Numerik")
    st.markdown("""
    Scatterplot berikut menunjukkan hubungan visual antara harga dan tiap fitur numerik.  
    Beberapa fitur seperti `curbweight`, `enginesize`, dan `horsepower` menunjukkan hubungan linier yang cukup kuat, 
    sementara fitur lain menyebar merata dan tetap relevan dalam model kompleks.
    """)
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_features.remove('price')
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(numerical_features):
        sns.scatterplot(x=df[feature], y=df['price'], ax=axes[i])
        axes[i].set_title(f'{feature} vs Price')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Price')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Hubungan antara Harga dan Fitur Numerik', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)

    # 4. Boxplot Harga per Brand
    st.markdown("### Harga Mobil Berdasarkan Merek")
    st.markdown("""
    Boxplot ini menunjukkan perbedaan harga antar merek.  
    Merek seperti **Jaguar**, **BMW**, dan **Buick** memiliki median harga tinggi, sedangkan merek seperti **Honda**, **Mazda**, dan **Nissan** berada di segmen harga rendah.  
    Terlihat pula adanya outlier dari beberapa brand, menunjukkan varian model spesifik dengan harga ekstrem.
    """)
    fig, ax = plt.subplots(figsize=(14, 6))
    order = df.groupby('CarBrand')['price'].median().sort_values().index
    sns.boxplot(data=df, x='CarBrand', y='price', order=order, ax=ax)
    plt.xticks(rotation=90)
    ax.set_title("Boxplot Harga per Brand")
    st.pyplot(fig)

# -----------------------------
# MENU: EVALUASI MODEL
# -----------------------------
elif menu == "Evaluasi Model":
    st.subheader("📈 Evaluasi Model")

    # ---------------------
    # Evaluasi: Random Forest
    # ---------------------
    st.markdown("### 📊 Regresi: Random Forest")

    X = df.drop(columns=['price'])
    y = df['price']
    X_encoded = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${mae:,.2f}")
    col2.metric("RMSE", f"${rmse:,.2f}")
    col3.metric("R²", f"{r2:.4f}")

    st.markdown("""
    Random Forest merupakan model dengan performa terbaik.  
    Dengan R² sebesar **0.9575**, model mampu menjelaskan hampir 96% variasi harga mobil, 
    menjadikannya sangat akurat dan andal.
    """)

    # ---------------------
    # Evaluasi: K-Means Clustering
    # ---------------------
    st.markdown("### 🔵 Clustering: K-Means")

    # Fitur numerik relevan untuk clustering harga
    fitur_cluster = ['price', 'horsepower', 'enginesize', 'curbweight', 'citympg', 'highwaympg']
    df_cluster = df[fitur_cluster].dropna()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_cluster)

    # Cari inertia untuk range K (optional, bisa divisualisasikan di menu eksplorasi)
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        inertias.append(kmeans.inertia_)

    # KMeans akhir dengan k=5
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster_Price'] = kmeans.fit_predict(data_scaled)

    # Urutkan klaster berdasarkan harga rata-rata
    cluster_means = df.groupby('Cluster_Price')['price'].mean().sort_values()
    ordered_clusters = cluster_means.index.tolist()
    cluster_labels = {
        ordered_clusters[0]: 'murah sekali',
        ordered_clusters[1]: 'murah',
        ordered_clusters[2]: 'sedang',
        ordered_clusters[3]: 'mahal',
        ordered_clusters[4]: 'sangat mahal'
    }

    df['Price_Label'] = df['Cluster_Price'].map(cluster_labels)

    # Hitung Silhouette Score
    sil_score = silhouette_score(data_scaled, df['Cluster_Price'])
    st.metric("Silhouette Score", f"{sil_score:.4f}")

    st.markdown(f"""
    KMeans clustering dengan **5 klaster** memberikan Silhouette Score sebesar **{sil_score:.4f}**.  
    Nilai ini berada pada kategori **cukup baik**, menunjukkan bahwa struktur data cukup mendukung untuk dibagi menjadi lima kelompok harga.  
    Setiap klaster telah diberi label seperti 'sangat murah', 'murah', 'sedang', 'mahal' hingga 'sangat mahal' untuk membantu interpretasi hasil klasterisasi.
    """)

# -----------------------------
# MENU: PREDIKSI HARGA MOBIL
# -----------------------------
elif menu == "Prediksi Harga Mobil":
    st.subheader("🔍 Prediksi Harga Mobil")

    with st.expander("📋 Input Spesifikasi Mobil", expanded=True):
        with st.form("form_input"):
            col1, col2 = st.columns(2)
            with col1:
                carbrand = st.selectbox("Car Brand", sorted(df_rekom['CarBrand'].dropna().unique()))
                fueltype = st.selectbox("Fuel Type", ['gas', 'diesel'])
                aspiration = st.selectbox("Aspiration", ['std', 'turbo'])
                carbody = st.selectbox("Car Body", ['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'])
                drivewheel = st.selectbox("Drive Wheel", ['fwd', 'rwd', '4wd'])
                enginelocation = st.selectbox("Engine Location", ['front', 'rear'])
                enginetype = st.selectbox("Engine Type", ['dohc', 'ohcv', 'ohc', 'rotor', 'ohcf', 'l', 'dohcv'])
                cylindernumber = st.selectbox("Cylinder Number", ['four', 'six', 'five', 'eight', 'two', 'three', 'twelve'])
                fuelsystem = st.selectbox("Fuel System", ['mpfi', '2bbl', '1bbl', 'spdi', '4bbl', 'idi', 'spfi'])
                symboling = st.slider("Symboling", -2, 3, 0)
            with col2:
                wheelbase = st.slider("Wheelbase", 85.0, 120.0, 100.0)
                curbweight = st.slider("Curb Weight", 1500, 4000, 2500)
                enginesize = st.slider("Engine Size", 60, 350, 130)
                boreratio = st.slider("Bore Ratio", 2.0, 4.0, 3.0)
                stroke = st.slider("Stroke", 2.0, 4.0, 3.0)
                compressionratio = st.slider("Compression Ratio", 7.0, 23.0, 10.0)
                horsepower = st.slider("Horsepower", 48, 288, 100)
                peakrpm = st.slider("Peak RPM", 4000, 6600, 5200)
                citympg = st.slider("City MPG", 10, 50, 25)
                highwaympg = st.slider("Highway MPG", 15, 60, 30)

            submitted = st.form_submit_button("🔍 Prediksi Harga")

    if submitted:
        data_input = pd.DataFrame([{
            'symboling': symboling,
            'CarBrand': carbrand.lower(),
            'fueltype': fueltype,
            'aspiration': aspiration,
            'doornumber': 'four',
            'carbody': carbody,
            'drivewheel': drivewheel,
            'enginelocation': enginelocation,
            'wheelbase': wheelbase,
            'carlength': 170.0,
            'carwidth': 65.0,
            'carheight': 52.0,
            'curbweight': curbweight,
            'enginetype': enginetype,
            'cylindernumber': cylindernumber,
            'enginesize': enginesize,
            'fuelsystem': fuelsystem,
            'boreratio': boreratio,
            'stroke': stroke,
            'compressionratio': compressionratio,
            'horsepower': horsepower,
            'peakrpm': peakrpm,
            'citympg': citympg,
            'highwaympg': highwaympg
        }])

        pred_harga = model_rf.predict(data_input)[0]

        fitur_cluster = np.array([[pred_harga, horsepower, enginesize, curbweight, citympg, highwaympg]])
        fitur_scaled = scaler_kmeans.transform(fitur_cluster)
        cluster = model_kmeans.predict(fitur_scaled)[0]
        kategori = cluster_label_map[str(cluster)]

        st.session_state.pred_harga = pred_harga
        st.session_state.kategori = kategori
        st.session_state.carbrand = carbrand

    if 'pred_harga' in st.session_state:
        col_left, col_right = st.columns([2, 2])
        with col_left:
            st.metric("💰 Estimasi Harga Mobil", f"${st.session_state.pred_harga:,.2f}")
            st.write(f"**Brand:** {st.session_state.carbrand.title()}")
        with col_right:
            st.metric("🏷️ Kategori Harga", st.session_state.kategori.upper())

        st.divider()
        if st.button("🚘 Tampilkan Rekomendasi Mobil Sejenis"):
            fitur_all = ['price', 'horsepower', 'enginesize', 'curbweight', 'citympg', 'highwaympg']
            try:
                data_scaled = scaler_kmeans.transform(df_rekom[fitur_all])
                clusters = model_kmeans.predict(data_scaled)
                df_rekom['Cluster_Price'] = clusters
                df_rekom['Price_Label'] = df_rekom['Cluster_Price'].map(lambda x: cluster_label_map[str(x)])

                hasil = df_rekom[df_rekom['Price_Label'] == st.session_state.kategori]
                hasil_tampil = hasil[['CarBrand', 'CarModel', 'price']].sort_values(by='price').head(5)

                st.subheader("🔁 Rekomendasi Mobil Sejenis:")
                st.dataframe(hasil_tampil.rename(columns={
                    'CarBrand': 'Brand',
                    'CarModel': 'Model',
                    'price': 'Harga ($)'
                }), use_container_width=True)
            except Exception as e:
                st.error(f"Terjadi error saat mencari rekomendasi: {e}")
