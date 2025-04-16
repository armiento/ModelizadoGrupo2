import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
import numpy as np

# -------------------------------
# ✅ Configuración general
# -------------------------------
st.set_page_config(
    page_title="Perro vs Gato Classificador", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern and clean look
st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background-color: #f8f9fa;
        padding: 1.5rem;
    }
    
    /* Header and title styling */
    h1 {
        color: #1e3a8a;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2, h3 {
        color: #1e3a8a;
        font-weight: 600 !important;
    }
    
    /* Card-like containers */
    .stFrame, div[data-testid="stVerticalBlock"] > div:has(img) {
        background-color: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        padding: 0.5em 1.2em;
        font-size: 1rem;
        font-weight: 500;
        background-color: #3b82f6;
        color: white;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #2563eb;
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.2);
        transform: translateY(-1px);
    }
    
    /* Delete button styling */
    button[kind="secondary"] {
        background-color: #ef4444 !important;
    }
    
    button[kind="secondary"]:hover {
        background-color: #dc2626 !important;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background-color: #f0f9ff;
        border-radius: 8px;
        border: 1px dashed #93c5fd;
    }
    
    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border-color: #e2e8f0;
    }
    
    /* Result highlight container */
    .result-container {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }
    
    /* Tab styling for the application sections */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 16px;
        box-shadow: 0px -2px 4px rgba(0, 0, 0, 0.02);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #dbeafe;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# ✅ Cargar modelo
# -------------------------------
@st.cache_resource
def load_model(path="best_model.keras"):
    if not os.path.exists(path):
        st.error("❌ Modelo no encontrado. Asegúrate de entrenarlo primero.")
        st.stop()
    return tf.keras.models.load_model(path)

# -------------------------------
# ✅ Cargar etiquetas
# -------------------------------
@st.cache_data
def load_labels(train_dir="dataset/test_set/test_set"):
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        st.error("❌ Directorio de datos no encontrado o vacío.")
        st.stop()
    return sorted(os.listdir(train_dir))

# -------------------------------
# ✅ Función para procesar imagen
# -------------------------------
def process_image(image):
    image_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    return img_array

# -------------------------------
# ✅ Función para mostrar resultados
# -------------------------------
def display_prediction_results(predictions, label_list, uploaded_file):
    top_indices = predictions.argsort()[-2:][::-1]
    
    pred_df = pd.DataFrame({
        "Animal": [label_list[idx].capitalize() for idx in top_indices],
        "Probabilidad (%)": predictions[top_indices] * 100
    })
    
    predicted_class = label_list[top_indices[0]].capitalize()
    confidence = predictions[top_indices[0]] * 100
    emoji = "🐶" if "dog" in predicted_class.lower() else "🐱"
    
    # Crear contenedor de resultados con estilo
    st.markdown(f"""
        <div class="result-container">
            <h3 style="color:#047857; margin-bottom:0.5rem;">🎯 Resultado: {predicted_class} {emoji}</h3>
            <p style="font-size:1.2rem; font-weight:500; color:#059669;">
                📊 Confianza: {confidence:.2f}%
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Gráfico mejorado
    fig = px.bar(
        pred_df, 
        x='Probabilidad (%)', 
        y='Animal',
        orientation='h',
        color='Animal',
        color_discrete_sequence=['#3b82f6', '#f97316'],
        text=[f'{val:.2f}%' for val in pred_df['Probabilidad (%)']],
        height=200
    )
    
    fig.update_layout(
        title='Probabilidades de la predicción',
        title_font_size=18,
        title_font_family="Arial",
        title_font_color="#1e3a8a",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            range=[0, 100]
        ),
        yaxis=dict(
            categoryorder='total ascending'
        )
    )
    
    fig.update_traces(
        textposition='auto',
        textfont_size=14,
        marker=dict(line=dict(width=1, color='#fff'))
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Agregar a historial
    if uploaded_file is not None:
        st.session_state.historial.append({
            "Archivo": uploaded_file.name,
            "Clase": predicted_class,
            "Confianza (%)": round(confidence, 2),
            "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Descargar resultado actual
    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar predicción (CSV)",
        data=csv,
        file_name="prediccion_actual.csv",
        mime="text/csv",
        key="download_current"
    )

# -------------------------------
# ✅ Mostrar historial mejorado
# -------------------------------
def display_history():
    if st.session_state.historial:
        st.markdown("## 📜 Historial de predicciones")
        
        historial_df = pd.DataFrame(st.session_state.historial)
        
        # Mejorar visualización del historial
        column_config = {
            "Archivo": st.column_config.TextColumn("📄 Archivo"),
            "Clase": st.column_config.TextColumn("🏷️ Clase"),
            "Confianza (%)": st.column_config.NumberColumn(
                "📊 Confianza",
                format="%.2f%%",
                help="Nivel de confianza de la predicción"
            ),
            "Fecha": st.column_config.DatetimeColumn(
                "🕒 Fecha y Hora",
                format="DD/MM/YYYY HH:mm"
            )
        }
        
        st.dataframe(
            historial_df,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
        
        # Opciones de historial en columnas
        col1, col2 = st.columns(2)
        
        with col1:
            # Descargar historial
            csv_hist = historial_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar historial completo",
                data=csv_hist,
                file_name="historial_predicciones.csv",
                mime="text/csv",
                key="download_history"
            )
        
        with col2:
            # Botón para borrar historial
            if st.button("🗑 Borrar historial", key="clear_history", type="secondary"):
                st.session_state.historial = []
                st.success("✅ Historial borrado correctamente")
                st.experimental_rerun()
    else:
        st.info("📝 Aún no hay predicciones en el historial")

# -------------------------------
# ✅ Mostrar estadísticas 
# -------------------------------
def display_statistics():
    if not st.session_state.historial:
        st.info("📊 No hay datos suficientes para mostrar estadísticas")
        return
    
    historial_df = pd.DataFrame(st.session_state.historial)
    
    st.markdown("## 📊 Estadísticas de las predicciones")
    
    # Métricas en tres columnas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total de predicciones", 
            len(historial_df),
            delta=None
        )
    
    with col2:
        perros = len(historial_df[historial_df["Clase"].str.lower() == "dogs"])
        st.metric(
            "Perros identificados", 
            perros,
            delta=f"{perros/len(historial_df)*100:.1f}%" if len(historial_df) > 0 else None
        )
    
    with col3:
        gatos = len(historial_df[historial_df["Clase"].str.lower() == "cats"])
        st.metric(
            "Gatos identificados", 
            gatos,
            delta=f"{gatos/len(historial_df)*100:.1f}%" if len(historial_df) > 0 else None
        )
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribución de clases")
        class_counts = historial_df["Clase"].value_counts().reset_index()
        class_counts.columns = ["Clase", "Cantidad"]
        
        fig_pie = px.pie(
            class_counts, 
            values="Cantidad", 
            names="Clase",
            color="Clase",
            color_discrete_map={"Dog": "#3b82f6", "Cat": "#f97316"},
            hole=0.4
        )
        
        fig_pie.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            legend_title="",
            legend=dict(orientation="h", y=-0.1)
        )
        
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont_size=12
        )
        
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
    
    with col2:
        st.markdown("### Confianza promedio por clase")
        
        avg_confidence = historial_df.groupby("Clase")["Confianza (%)"].mean().reset_index()
        
        fig_bar = px.bar(
            avg_confidence,
            x="Clase",
            y="Confianza (%)",
            color="Clase",
            color_discrete_map={"Dog": "#3b82f6", "Cat": "#f97316"},
            text_auto=".2f"
        )
        
        fig_bar.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            yaxis=dict(range=[0, 100], title="Confianza promedio (%)")
        )
        
        fig_bar.update_traces(
            textfont_size=12,
            textangle=0,
            textposition="outside",
            cliponaxis=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
    
    # Histograma de confianza
    st.markdown("### Distribución de niveles de confianza")
    
    fig_hist = px.histogram(
        historial_df,
        x="Confianza (%)",
        color="Clase",
        nbins=10,
        color_discrete_map={"Dog": "#3b82f6", "Cat": "#f97316"},
        barmode="overlay",
        opacity=0.7
    )
    
    fig_hist.update_layout(
        xaxis_title="Nivel de confianza (%)",
        yaxis_title="Frecuencia",
        legend_title="",
        legend=dict(orientation="h", y=1.1)
    )
    
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

# -------------------------------
# ✅ App principal
# -------------------------------
def streamlit_dog_cat_classifier():
    # Header
    st.markdown("<h1 style='text-align: center;'>🐶🐱 Clasificador de Perros y Gatos</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Sube una imagen y el modelo te dirá si es un perro o un gato con inteligencia artificial</h3>", unsafe_allow_html=True)

    st.markdown("---")

    # Inicializar historial en session state
    if "historial" not in st.session_state:
        st.session_state.historial = []
    
    # Cargar modelo y etiquetas
    with st.spinner("🔄 Cargando modelo..."):
        model = load_model()
        label_list = load_labels()
    
    # Crear pestañas para organizar la interfaz
    tab1, tab2, tab3 = st.tabs(["🔍 Clasificación", "📜 Historial", "📊 Estadísticas"])
    
    # Pestaña de clasificación
    with tab1:
        st.markdown("## 📸 Sube una imagen para clasificar")
        
        # Contenedor para imagen de muestra
        sample_container = st.container()
        sample_container.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f3f4f6; border-radius: 10px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: center; gap: 20px;">
                <div style="text-align: center;">
                    <img src="https://cdn.pixabay.com/photo/2016/03/28/12/35/cat-1285634_640.png" width="120" 
                    width="120" height="120"
                    style="border-radius: 10px; margin-bottom: 10px; object-fit: cover;">
                    <p style="margin: 0; font-weight: 500;">Gato 🐱</p>
                </div>
                <div style="text-align: center;">
                    <img src="https://thumbs.dreamstime.com/b/two-month-old-shih-tzu-dog-adorable-puppy-372065032.jpg?w=576"  
                    width="120" height="120"
                    style="border-radius: 10px; margin-bottom: 10px; object-fit: cover;">
                    <p style="margin: 0; font-weight: 500;">Perro 🐶</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload de imagen
        uploaded_file = st.file_uploader(
            "📤 Por favor, que tu imagen sea clara y de un solo animal", 
            type=["jpg", "jpeg", "png"],
            help="Sube una imagen con una buena iluminación para obtener mejores resultados"
        )
        
        # Sección de clasificación
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                
                # Mostrar imagen y proceso de predicción
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### 🖼️ Imagen cargada")
                    st.image(image, caption=f"Archivo: {uploaded_file.name}", use_container_width=True)
                
                with col2:
                    st.markdown("### 🧠 Análisis con IA")
                    with st.spinner("🔍 Analizando imagen..."):
                        # Simular un pequeño retraso para mejor experiencia visual
                        import time
                        time.sleep(0.5)
                        
                        # Procesar imagen y predecir
                        img_array = process_image(image)
                        predictions = model.predict(img_array)[0]
                        
                        # Mostrar resultados
                        display_prediction_results(predictions, label_list, uploaded_file)
            except Exception as e:
                st.error(f"❌ Error al procesar la imagen: {str(e)}")
                st.info("💡 Intenta con otra imagen o asegúrate de que el formato sea correcto.")
        else:
            # Mensaje cuando no hay imagen
            st.info("👆 Sube una imagen de un perro o un gato para comenzar")
    
    # Pestaña de historial
    with tab2:
        display_history()
    
    # Pestaña de estadísticas
    with tab3:
        display_statistics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 10px; color: #64748b; font-size: 0.8rem;">
            Desarrollado con ❤️ usando Streamlit, TensorFlow y Plotly | © 2025
        </div>
    """, unsafe_allow_html=True)

# -------------------------------
# ✅ Ejecutar app
# -------------------------------
if __name__ == "__main__":
    streamlit_dog_cat_classifier()