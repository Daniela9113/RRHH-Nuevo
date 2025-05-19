import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(layout="wide")

# Hack CSS para ajustar ancho mínimo de columnas en tablas
st.markdown("""
    <style>
    .stDataFrame table {
        min-width: 1000px;
    }
    th, td {
        white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)


# Título de la aplicación
st.title("Evaluación de Candidatos y Probabilidad de Contratación")

# Carga del modelo entrenado
@st.cache_resource
def load_model():
    return joblib.load('modelo_entrenado.pkl')

model = load_model()

# Subida del dataset
uploaded_file = st.file_uploader("Sube tu nuevo dataset (CSV)", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Por favor, sube un archivo CSV para continuar.")
    st.stop()

# Columnas usadas en el entrenamiento
categorical_cols = ["educacion", "experiencia_sector", "nivel_ingles"]
numerical_cols = ["experiencia_anios", "certificaciones", "puntaje_test", "puntaje_entrevista", "referencia_interna"]
all_cols = categorical_cols + numerical_cols

# Validación
def validate_features(df, cols):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        st.error(f"Faltan columnas requeridas en el CSV: {', '.join(missing)}")
        st.stop()

validate_features(df, all_cols)

# Codificación como en entrenamiento
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df[categorical_cols])
encoded_cat_df = pd.DataFrame(encoder.transform(df[categorical_cols]), columns=encoder.get_feature_names_out())

# Concatenación
X = pd.concat([encoded_cat_df, df[numerical_cols].reset_index(drop=True)], axis=1)

# Predicción
df["probabilidad_contratacion"] = model.predict_proba(X)[:, 1]
df["prediccion_contratacion"] = model.predict(X)

# Mostrar tabla general
st.subheader("Resultados de Predicciones")
st.dataframe(df)

# Validar existencia de columna vacante_id
if 'vacante_id' not in df.columns:
    st.warning("La columna 'vacante_id' no está en el dataset. No se puede continuar con el análisis por vacante.")
    st.stop()

# Selección de vacante
vacantes = df['vacante_id'].unique().tolist()
vacante_seleccionada = st.selectbox("Selecciona Vacante", vacantes)

# Filtrar y ordenar
df_filtrado = df[df['vacante_id'] == vacante_seleccionada].sort_values(
    'probabilidad_contratacion', ascending=False
)

# Gráfico interactivo
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=df_filtrado.index.astype(str),
        y=df_filtrado['probabilidad_contratacion'],
        hovertemplate=
            "<b>Candidato</b>: %{x}<br>" +
            "<b>Probabilidad</b>: %{y:.2%}<br>" +
            "<b>Educación</b>: %{customdata[0]}<br>" +
            "<b>Años de Experiencia</b>: %{customdata[1]}<br>" +
            "<b>Sector</b>: %{customdata[2]}<br>" +
            "<extra></extra>",
        customdata=df_filtrado[['educacion', 'experiencia_anios', 'experiencia_sector']].values
    )
)
fig.update_layout(
    title=f"Probabilidades de Contratación - Vacante {vacante_seleccionada}",
    xaxis_title="Índice de Candidato",
    yaxis_title="Probabilidad de Contratación",
    margin=dict(t=50)
)

st.plotly_chart(fig)

# Explicación del gráfico
st.markdown(
    "**Gráfico de Barras**: Muestra la probabilidad de contratación de cada candidato para la vacante seleccionada. "
    "Los candidatos están ordenados de mayor a menor probabilidad, facilitando la identificación de los mejores perfiles."
)

# Slider Top N
top_n = st.slider(
    "Número de candidatos a mostrar", min_value=1, max_value=len(df_filtrado), value=min(10, len(df_filtrado))
)

# Tabla Top N con nombres
tabla_top = df_filtrado.head(top_n)[[
    'probabilidad_contratacion',
    'prediccion_contratacion',
    'educacion',
    'experiencia_anios',
    'experiencia_sector'
]].rename(columns={
    'probabilidad_contratacion': 'Prob.Contrat',
    'prediccion_contratacion': 'Predicción',
    'educacion': 'Educación',
    'experiencia_anios': 'Años Exp.',
    'experiencia_sector': 'Sector'
})

st.subheader(f"Top {top_n} candidatos por probabilidad")
st.dataframe(tabla_top, use_container_width=True)

# Comentario final
st.markdown(
    "**Tabla de los Top Candidatos**: Presenta los candidatos mejor posicionados según la probabilidad de contratación, "
    "junto con sus características clave."
)

