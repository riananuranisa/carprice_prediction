import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans

# === Load Dataset ===
df = pd.read_csv("CarPrice_Assignment.csv")

# Pisahkan brand
df[['CarBrand', 'CarModel']] = df['CarName'].str.split(' ', n=1, expand=True)
brand_corrections = {
    'toyouta': 'toyota',
    'Nissan': 'nissan',
    'maxda': 'mazda',
    'vw': 'volkswagen',
    'vokswagen': 'volkswagen',
    'porcshce': 'porsche',
}
df['CarBrand'] = df['CarBrand'].replace(brand_corrections).str.lower()

# Drop kolom tidak relevan
df.drop(columns=['car_ID', 'CarName', 'CarModel'], inplace=True)

# === Build Random Forest ===
X = df.drop(columns=['price'])
y = df['price']

categorical_cols = X.select_dtypes(include='object').columns.tolist()
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)

print("📊 Evaluasi Random Forest")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# Simpan model Random Forest
with open("model_rf.pkl", "wb") as f:
    pickle.dump(model_rf, f)

# === Build KMeans Clustering ===
fitur_cluster = ['price', 'horsepower', 'enginesize', 'curbweight', 'citympg', 'highwaympg']
df_cluster = df[fitur_cluster].dropna()

scaler_kmeans = StandardScaler()
data_scaled = scaler_kmeans.fit_transform(df_cluster)

model_kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = model_kmeans.fit_predict(data_scaled)
df['Cluster_Price'] = cluster_labels

# Mapping ke kategori harga
cluster_means = df.groupby('Cluster_Price')['price'].mean().sort_values()
ordered_clusters = cluster_means.index.tolist()
price_label_mapping = {
    ordered_clusters[0]: 'murah sekali',
    ordered_clusters[1]: 'murah',
    ordered_clusters[2]: 'sedang',
    ordered_clusters[3]: 'mahal',
    ordered_clusters[4]: 'sangat mahal'
}
df['Price_Label'] = df['Cluster_Price'].map(price_label_mapping)

# Simpan KMeans dan Scaler
with open("model_kmeans.pkl", "wb") as f:
    pickle.dump(model_kmeans, f)

with open("scaler_kmeans.pkl", "wb") as f:
    pickle.dump(scaler_kmeans, f)

with open("cluster_labels.json", "w") as f:
    json.dump(price_label_mapping, f)

print("✅ Semua model berhasil dibuat dan disimpan.")
