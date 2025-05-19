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

# Título
st.title("Evaluación de Candidatos y Probabilidad de Contratación")

# Cargar modelo
@st.cache_resource
def load_model():
    return joblib.load('modelo_entrenado.pkl')

model = load_model()

# Subida del CSV
uploaded_file = st.file_uploader("Sube tu nuevo dataset (CSV)", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Por favor, sube un archivo CSV para continuar.")
    st.stop()

# Columnas necesarias
categorical_cols = ["educacion", "experiencia_sector", "nivel_ingles"]
numerical_cols = ["experiencia_anios", "certificaciones", "puntaje_test", "puntaje_entrevista", "referencia_interna"]
all_cols = categorical_cols + numerical_cols

# Validar columnas
def validate_features(df, cols):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        st.error(f"Faltan columnas requeridas en el CSV: {', '.join(missing)}")
        st.stop()

validate_features(df, all_cols)

# Codificar variables categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df[categorical_cols])
encoded_cat_df = pd.DataFrame(encoder.transform(df[categorical_cols]), columns=encoder.get_feature_names_out())

# Preparar X
X = pd.concat([encoded_cat_df, df[numerical_cols].reset_index(drop=True)], axis=1)

# Predicción
df["probabilidad_contratacion"] = model.predict_proba(X)[:, 1]
df["prediccion_contratacion"] = model.predict(X)

# Mostrar tabla completa
st.subheader("Resultados de Predicciones")
st.dataframe(df)

# Validar columna de vacante
if 'vacante_id' not in df.columns:
    st.warning("La columna 'vacante_id' no está en el dataset. No se puede continuar con el análisis por vacante.")
    st.stop()

# Selección de vacante
vacantes = df['vacante_id'].unique().tolist()
vacante_seleccionada = st.selectbox("Selecciona Vacante", vacantes)

# Filtrar por vacante
df_filtrado = df[df['vacante_id'] == vacante_seleccionada].sort_values('probabilidad_contratacion', ascending=False)

# Crear identificador para los candidatos
df_filtrado['candidato'] = 'Candidato ' + df_filtrado.index.astype(str)

# Colorear barras según probabilidad
colors = df_filtrado['probabilidad_contratacion'].apply(
    lambda x: 'green' if x > 0.7 else 'orange' if x > 0.4 else 'red'
)

# Etiquetas sobre las barras
text_labels = df_filtrado['probabilidad_contratacion'].apply(lambda x: f"{x:.0%}")

# Gráfico mejorado
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=df_filtrado['probabilidad_contratacion'],
        y=df_filtrado['candidato'],
        orientation='h',
        marker_color=colors,
        text=text_labels,
        textposition='auto',
        customdata=df_filtrado[['educacion', 'experiencia_anios', 'experiencia_sector']],
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "<b>Probabilidad:</b> %{x:.2%}<br>" +
            "<b>Educación:</b> %{customdata[0]}<br>" +
            "<b>Años de Experiencia:</b> %{customdata[1]}<br>" +
            "<b>Sector:</b> %{customdata[2]}<br>" +
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title=f"Probabilidades de Contratación - Vacante {vacante_seleccionada}",
    xaxis_title="Probabilidad de Contratación",
    yaxis_title="Candidatos",
    margin=dict(t=60, l=100),
    height=500,
    yaxis=dict(autorange="reversed")
)

# Mostrar gráfico
st.plotly_chart(fig, use_container_width=True)

# Descripción del gráfico
st.markdown(
    "**Gráfico de Barras**: Muestra la probabilidad de contratación de cada candidato para la vacante seleccionada. "
    "Los candidatos están ordenados de mayor a menor probabilidad, facilitando la identificación de los mejores perfiles."
)

# Slider para Top N
top_n = st.slider(
    "Número de candidatos a mostrar", min_value=1, max_value=len(df_filtrado), value=min(10, len(df_filtrado))
)

# Tabla con Top N candidatos
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
