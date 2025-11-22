import pandas as pd
import plotly.express as px
import streamlit as st
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Cargar dataset Iris
iris = load_iris(as_frame=True)
df = iris.frame.copy()

df = df.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width"
})

df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

# 2. Modelo de clasificación
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(random_state=42)
model.fit(X, y_encoded)

# 3. INTERFAZ
# sliders para las medidas
st.sidebar.header("Ingrese las medidas de su flor")
sepal_length = st.sidebar.slider("Longitud del sépalo (cm)", 3.0, 9.0, 5.5, 0.1)
sepal_width = st.sidebar.slider("Ancho del sépalo (cm)", 1.5, 5.0, 2.8, 0.1)
petal_length = st.sidebar.slider("Longitud del pétalo (cm)", 0.5, 7.0, 5.0, 0.1)
petal_width = st.sidebar.slider("Ancho del pétalo (cm)", 0.1, 2.5, 1.8, 0.1)

# Construir DataFrame de la flor
mi_flor = pd.DataFrame([{
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width
}])

# Predicción
pred_encoded = model.predict(mi_flor)[0]
pred_species = le.inverse_transform([pred_encoded])[0]

# Probabilidades
probs = model.predict_proba(mi_flor)[0]
proba_dict = {especie: float(p) for especie, p in zip(le.classes_, probs)}
proba_df = pd.DataFrame.from_dict(proba_dict, orient='index', columns=['probability'])
proba_df['probability_pct'] = proba_df['probability'].map(lambda x: f"{x:.2%}")

# Añadir especie al dataframe para graficar
mi_flor["species"] = "Tu flor"


# 4. Union de dataset con resultado
df_plot = pd.concat([df, mi_flor], ignore_index=True)
df_plot["origen"] = ["dataset"] * len(df) + ["tu_flor"]

df_plot["point_size"] = df_plot["origen"].map({
    "dataset": 7,
    "tu_flor": 18
})

# 5. Gráfica 3D
color_map = {
    "setosa": "#1f77b4",  
    "versicolor": "#FFD700", 
    "virginica": "#2ca02c",  
    "Tu flor": "#FF0000"     
}

fig = px.scatter_3d(
    df_plot,
    x="sepal_length",
    y="sepal_width",
    z="petal_length",
    color="species",
    size="point_size",
    symbol="species",  
    title=f"     Tu flor se parece a: {pred_species}",
    hover_data=["sepal_length", "sepal_width", "petal_length", "petal_width", "origen"],
    color_discrete_map=color_map,
    category_orders={"species": ["setosa", "versicolor", "virginica", "Tu flor"]},
    opacity=0.85,
    size_max=22
)

# Aumentar contraste de los puntos
fig.update_traces(marker=dict(line=dict(width=0.6, color='DarkSlateGrey')))
fig.update_layout(scene_aspectmode="cube", legend=dict(title_text='Especies'))

# Mostrar la figura en Streamlit
st.plotly_chart(fig, use_container_width=True)

# 6. Mostrar imagen 
st.subheader("Posible imagen según la predicción")
image_map = {
    "setosa": "setosa.jpg",
    "versicolor": "versicolor.jpg",
    "virginica": "virginica.jpg"
}

ruta = image_map.get(pred_species)
if ruta and os.path.exists(ruta):
    st.image(ruta, caption=f"Predicción: {pred_species}", use_container_width=True)
else:
    st.warning(f"No se encontró la imagen para '{pred_species}'. Asegúrate de que {ruta} exista en el directorio del proyecto.")

# Mostrar probabilidad
st.subheader("Probabilidades")
st.table(proba_df[['probability_pct']].rename(columns={'probability_pct': 'probabilidad'}))
