import plotly.express as px
import pandas as pd
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



# css
st.set_page_config(page_title="Clasificador Iris 3D", page_icon="🌸", layout="wide")

st.markdown("""
<style>
body {background: #f3f4f6;}
/* Container padding */
.block-container{padding-top:0.4rem;padding-left:0.9rem;padding-right:0.9rem;padding-bottom:0.4rem}
.big-font {font-size: 1.85em; font-weight: 700; color: #1f77b4;}
.iris-header {
    background: linear-gradient(90deg, #e0e7ff 0%, #f0fdfa 100%);
    border-radius: 0 0 10px 10px;
    padding: 0.4em 0 0.2em 0;
    margin-bottom: 0.25em;
    box-shadow: 0 1px 6px rgba(31,119,180,0.04);
    text-align: center;
}
.card {
    background-color: #f8fafc;
    border-radius: 8px;
    padding: 0.45em 0.6em 0.6em 0.6em;
    margin-bottom: 0.25em;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
/* Reduce spacing inside markdown blocks */
.stMarkdown p{margin:0;padding:0}
.stExpander{margin:0;padding:0}
.badge {
    display: inline-block;
    padding: 0.22em 0.5em;
    font-size: 0.9em;
    font-weight: 600;
    border-radius: 1em;
    color: white;
    margin-bottom: 0.2em;
}
.badge-setosa {background: #1f77b4;}
.badge-versicolor {background: #FFD700; color: #222;}
.badge-virginica {background: #2ca02c;}
.badge-tuflor {background: #FF0000;}
.footer {
    color: #888; font-size: 0.82em; text-align: center; margin-top: 0.4em;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("🔧 Ajusta las medidas de tu flor")
    sepal_length = st.slider("Longitud del sépalo (cm)", 3.0, 9.0, 5.5, 0.1)
    sepal_width = st.slider("Ancho del sépalo (cm)", 1.5, 5.0, 2.8, 0.1)
    petal_length = st.slider("Longitud del pétalo (cm)", 0.5, 7.0, 5.0, 0.1)
    petal_width = st.slider("Ancho del pétalo (cm)", 0.1, 2.5, 1.8, 0.1)

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


# columnas 
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    color_map = {
        "setosa": "#1f77b4",  # azul
        "versicolor": "#FFD700", # amarillo
        "virginica": "#2ca02c",  # verde
        "Tu flor": "#FF0000"     # rojo
    }
    fig = px.scatter_3d(
        df_plot,
        x="sepal_length",
        y="sepal_width",
        z="petal_length",
        color="species",
        size="point_size",
        symbol="species",
        title=f" Tu flor se parece a:    {pred_species}",
        hover_data=["sepal_length", "sepal_width", "petal_length", "petal_width", "origen"],
        color_discrete_map=color_map,
        category_orders={"species": ["setosa", "versicolor", "virginica", "Tu flor"]},
        opacity=0.85,
        size_max=22
    )
    fig.update_traces(marker=dict(line=dict(width=0.6, color='DarkSlateGrey')))
    fig.update_layout(scene_aspectmode="cube", legend=dict(title_text='Especies'), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Tarjeta de predicción
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌼 Predicción de especie")
    badge_class = f"badge badge-{pred_species.lower() if pred_species in ['setosa','versicolor','virginica'] else 'tuflor'}"
    st.markdown(f"<span class='{badge_class}'>{pred_species.upper()}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tarjeta de probabilidades
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Probabilidades")
    for especie, p in zip(le.classes_, probs):
        color = {'setosa':'#1f77b4','versicolor':'#FFD700','virginica':'#2ca02c'}.get(especie,'#888')
        st.markdown(f"<div style='display:flex;align-items:center;gap:0.7em;margin-bottom:0.3em;'>"
                    f"<span style='min-width:90px;font-weight:600;color:{color};'>{especie.capitalize()}</span>"
                    f"<span style='min-width:55px;font-family:monospace;'>{p:.2%}</span>"
                    f"<div style='flex:1;'><div style='background:#e5e7eb;border-radius:8px;height:14px;overflow:hidden;'><div style='width:{p*100:.1f}%;background:{color};height:100%;'></div></div></div>"
                    f"</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Imagen pequeña 
    image_map = {
        "setosa": "setosa.jpg",
        "versicolor": "versicolor.jpg",
        "virginica": "virginica.jpg"
    }
    ruta = image_map.get(pred_species)
    if ruta and os.path.exists(ruta):
        spacer, img_col = st.columns([0.25, 0.75])
        with img_col:
            st.image(ruta, width=180, caption=f"{pred_species}")
    else:
        st.info("Imagen local no encontrada. Coloca 'setosa.jpg', 'versicolor.jpg' y 'virginica.jpg' en el directorio del proyecto.")



# Pie de página profesional
st.markdown('<div class="footer">Desarrollado por <b>JCertain</b> · Proyecto Iris · 2025 · <a href="https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html" target="_blank">Dataset Iris</a></div>', unsafe_allow_html=True)




