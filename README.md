# 🌸 IRIS SPECIES CLASSIFICATION  
### Proyecto Final – Data Mining  
Universidad de la Costa – CUC  
Profesor: **José Escorcia-Gutiérrez, Ph.D.**

---

## 📌 Descripción del proyecto

Este proyecto implementa un modelo de **clasificación de especies Iris** utilizando técnicas de *Data Mining* y un *dashboard interactivo* creado con **Streamlit**.

El objetivo es predecir la especie de una flor a partir de cuatro mediciones:

- Longitud del sépalo  
- Ancho del sépalo  
- Longitud del pétalo  
- Ancho del pétalo  

Se entrenó un modelo de **Random Forest**, se evaluó con métricas estándar y se diseñó un dashboard donde el usuario puede modificar las características de una flor y visualizar su predicción en **3D** junto a todo el dataset.

---

## 🎯 Objetivos del trabajo

✔ Integrar los conocimientos del curso en un proyecto end-to-end  
✔ Aplicar un workflow completo de minería de datos  
✔ Justificar las técnicas utilizadas para la clasificación  
✔ Diseñar un dashboard interactivo que muestre resultados y análisis  

---

## 🧠 Workflow del proyecto

### **1. Understanding (Comprensión del dataset)**
- Exploración del dataset Iris  
- Análisis de distribución por especie  
- Histogramas, scatter plots y matriz de relaciones  

### **2. Preprocesamiento**
- Renombramiento de columnas  
- Conversión de etiquetas numéricas a categorías  
- Codificación de etiquetas con `LabelEncoder`  

### **3. Modelado**
- Entrenamiento del modelo **RandomForestClassifier**  
- Selección del algoritmo por su estabilidad, precisión y bajo sobreajuste  
- Entrenamiento con las 4 características numéricas  

### **4. Evaluación**
Se calcularon las métricas recomendadas:

- **Accuracy**  
- **Precision (weighted)**  
- **Recall (weighted)**  
- **F1-score (weighted)**  

### **5. Dashboard Interactivo (Streamlit)**
El dashboard incluye:

🔹 Pestaña **Clasificador 3D**  
- Sliders para ingresar las medidas de una flor  
- Predicción de la especie  
- Barras de probabilidad  
- Imagen de la especie predicha  
- Gráfica 3D con dataset + punto de usuario  

🔹 Pestaña **Métricas del modelo**  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Explicación de cada métrica  

🔹 Pestaña **Exploración del dataset**  
- Histogramas por especie  
- Scatter plots  
- Vista previa del dataset  

---

## 🛠️ Tecnologías utilizadas

- Python  
- Scikit-Learn  
- Pandas  
- Plotly  
- Streamlit  
- Markdown  

---

## 📦 Instalación

Clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd nombre-del-repo
