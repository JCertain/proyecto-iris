import plotly.express as px
import pandas as pd
import streamlit as st
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss  # métricas

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

rf = RandomForestClassifier(random_state=42)
model = CalibratedClassifierCV(rf, cv=5)
model.fit(X, y_encoded)

# 3. Métricas del modelo (sobre todo el dataset)
y_pred_train = model.predict(X)

accuracy = accuracy_score(y_encoded, y_pred_train)
precision = precision_score(y_encoded, y_pred_train, average='weighted')
recall = recall_score(y_encoded, y_pred_train, average='weighted')
f1 = f1_score(y_encoded, y_pred_train, average='weighted')
logloss = log_loss(y_encoded, model.predict_proba(X))


# STREAMLIT UI 

st.set_page_config(page_title="Clasificador Iris 3D", page_icon=None, layout="wide")

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


# Encabezado visual antes de las tabs
st.markdown('<div class="iris-header" style="margin-top:2.5em;"><span class="big-font">Clasificador de Flores Iris</span><br><span style="font-size:1.15em; color:#444;">Explora el dataset Iris, predice la especie y visualiza el resultado en 3D.</span></div>', unsafe_allow_html=True)

# Tabs principales
tab1, tab2, tab3 = st.tabs([
    "Clasificador 3D",
    "Métricas del modelo",
    "Exploración del dataset"
])

# ================== TAB 1: CLASIFICADOR ==================
with tab1:
    with st.sidebar:
        st.header("Ajusta las medidas de tu flor")
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

    # Unión de dataset con resultado
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
            "setosa": "#1f77b4",      # azul
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
            title=f"Tu flor se parece a: {pred_species}",
            hover_data=["sepal_length", "sepal_width", "petal_length", "petal_width", "origen"],
            color_discrete_map=color_map,
            category_orders={"species": ["setosa", "versicolor", "virginica", "Tu flor"]},
            opacity=0.85,
            size_max=22
        )
        fig.update_traces(marker=dict(line=dict(width=0.6, color='DarkSlateGrey')))
        fig.update_layout(
            scene_aspectmode="cube",
            legend=dict(title_text='Especie'),
            scene = dict(
                xaxis_title='Longitud del sépalo (cm)',
                yaxis_title='Ancho del sépalo (cm)',
                zaxis_title='Longitud del pétalo (cm)'
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Tarjeta de predicción
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Predicción de especie")
        badge_class = f"badge badge-{pred_species.lower() if pred_species in ['setosa','versicolor','virginica'] else 'tuflor'}"
        st.markdown(f"<span class='{badge_class}'>{pred_species.upper()}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

#Posibilidades
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Probabilidades")
        sorted_probs = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
        pred_prob = proba_dict.get(pred_species, 0.0)
        st.markdown(f"<div style='font-weight:600;margin-bottom:0.35em;'>Confianza en la predicción: {pred_prob*100:.2f}</div>", unsafe_allow_html=True)
        for especie, p in sorted_probs:
            color = {'setosa':'#1f77b4','versicolor':'#FFD700','virginica':'#2ca02c'}.get(especie,'#888')
            highlight_style = "border:2px solid #1f77b4;padding:0.25em;border-radius:6px;" if especie == pred_species else ""
            label = f"<strong>{especie.capitalize()}</strong>" if especie == pred_species else especie.capitalize()
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.7em;margin-bottom:0.4em;{highlight_style}'>"
                f"<span style='min-width:110px;font-weight:600;color:{color};'>{label}</span>"
                f"<span style='min-width:75px;font-family:monospace;'>{p*100:.2f}</span>"
                f"<div style='flex:1;'><div style='background:#e5e7eb;border-radius:8px;height:14px;overflow:hidden;'>"
                f"<div style='width:{p*100:.2f}%;background:{color};height:100%;'></div>"
                f"</div></div>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Imagen
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


#  TAB 2: MÉTRICAS
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Métricas globales del modelo")

    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision (ponderada):** {precision:.4f}")
    st.write(f"**Recall (ponderado):** {recall:.4f}")
    st.write(f"**F1-score (ponderado):** {f1:.4f}")
    st.write(f"**Log-loss (calidad de probabilidades):** {logloss:.4f}")

    st.markdown("""
    Estas métricas se calcularon usando todo el dataset Iris:
    - **Accuracy** indica el porcentaje de clasificaciones correctas.
    - **Precision** mide qué tan precisas son las predicciones para cada clase.
    - **Recall** indica qué tanto recupera el modelo de los casos reales.
    - **F1-score** balancea precisión y recall en una sola métrica.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ================== TAB 3: EXPLORACIÓN DEL DATASET ==================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Exploración del dataset Iris")
    st.markdown(
        "Aquí se muestran visualizaciones adicionales para entender mejor la distribución "
        "de las características y cómo se separan las especies."
    )

    # Vista rápida del dataset
    st.markdown("#### Vista previa del dataset")
    st.dataframe(df)

    st.markdown("#### 📈 Histogramas por característica y especie")

    # Primera fila de histogramas
    c1, c2 = st.columns(2)
    with c1:
        fig_h1 = px.histogram(
            df,
            x="sepal_length",
            color="species",
            barmode="overlay",
            opacity=0.7,
            marginal="box",
            title="Distribución de la longitud del sépalo",
            labels={"sepal_length": "Longitud del sépalo (cm)", "species": "Especie"}
        )
        st.plotly_chart(fig_h1, use_container_width=True)

    with c2:
        fig_h2 = px.histogram(
            df,
            x="sepal_width",
            color="species",
            barmode="overlay",
            opacity=0.7,
            marginal="box",
            title="Distribución del ancho del sépalo",
            labels={"sepal_width": "Ancho del sépalo (cm)", "species": "Especie"}
        )
        st.plotly_chart(fig_h2, use_container_width=True)

    # Segunda fila de histogramas
    c3, c4 = st.columns(2)
    with c3:
        fig_h3 = px.histogram(
            df,
            x="petal_length",
            color="species",
            barmode="overlay",
            opacity=0.7,
            marginal="box",
            title="Distribución de la longitud del pétalo",
            labels={"petal_length": "Longitud del pétalo (cm)", "species": "Especie"}
        )
        st.plotly_chart(fig_h3, use_container_width=True)

    with c4:
        fig_h4 = px.histogram(
            df,
            x="petal_width",
            color="species",
            barmode="overlay",
            opacity=0.7,
            marginal="box",
            title="Distribución del ancho del pétalo",
            labels={"petal_width": "Ancho del pétalo (cm)", "species": "Especie"}
        )
        st.plotly_chart(fig_h4, use_container_width=True)

    st.markdown("#### Relación entre características")
    fig_scatter = px.scatter(
        df,
        x="sepal_length",
        y="petal_length",
        color="species",
    title="Relación entre longitud del sépalo y del pétalo",
    labels={"sepal_length": "Longitud del sépalo (cm)", "petal_length": "Longitud del pétalo (cm)", "species": "Especie"},
    hover_data=["sepal_width", "petal_width"]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# Pie de página profesional
st.markdown(
    '<div class="footer">Desarrollado por <b>JCertain</b> · Proyecto Iris · 2025 · '
    '<a href="https://www.kaggle.com/datasets/uciml/iris?resource=download" target="_blank">'
    'Dataset Iris</a></div>',
    unsafe_allow_html=True
)
