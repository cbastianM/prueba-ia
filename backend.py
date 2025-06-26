# -------------------
# LIBRERÍAS (Versión Simplificada)
# -------------------
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import re

# -------------------
# CONFIGURACIÓN DE LA PÁGINA Y API
# -------------------
st.set_page_config(page_title="Profesor de Ing. Civil", page_icon="🎓")
st.title("🎓 Profesor Virtual de Ingeniería Civil")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.error("⚠️ No se encontró la GEMINI_API_KEY.")
    st.stop()

# -------------------
# CARGA DE DATOS DESDE CSV EN GITHUB
# -------------------
@st.cache_data(ttl=3600)
def load_knowledge_base():
    """Lee el CSV desde una URL fija de GitHub."""
    csv_url = "https://raw.githubusercontent.com/cbastianM/prueba-ia/main/Conocimiento_Ing_Civil.csv"
    try:
        df = pd.read_csv(csv_url, skipinitialspace=True)
        df.fillna('', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Error al leer el archivo CSV desde GitHub: {e}")
        return None

# -------------------
# LÓGICA DEL CHATBOT (SIMPLIFICADA)
# -------------------
def generate_response(query, dataframe):
    """
    Genera una respuesta de texto y una cadena Markdown para mostrar la imagen.
    """
    # Limpieza y búsqueda de la fila
    query_cleaned = query.strip().lower()
    match_df = dataframe[dataframe['ID_Ejercicio'].str.strip().str.lower() == query_cleaned]
    
    if match_df.empty:
        return f"Lo siento, no pude encontrar el ejercicio '{query}'. Por favor, asegúrate de escribir el ID exacto (ej: 'ejercicio 1.1').", ""

    match_row = match_df.iloc[0]

    # --- LÓGICA DE IMAGEN CON MARKDOWN ---
    context_text = match_row['Contenido']
    image_url = match_row['URL_Imagen']
    image_markdown = "" # String vacío por defecto
    
    if image_url and isinstance(image_url, str):
        # Limpiamos la URL por si acaso
        cleaned_url = image_url.strip().strip("'\"")
        # Creamos la cadena de Markdown para la imagen
        image_markdown = f"![Diagrama del Ejercicio]({cleaned_url})"
            
    # --- GENERACIÓN DE LA RESPUESTA DE LA IA (SOLO TEXTO) ---
    model = genai.GenerativeModel('gemini-1.5-flash-latest') # Usamos un modelo más rápido y económico
    
    prompt = f"""
    **ROL:** Eres un profesor de Ingeniería Civil.
    **TAREA:** Explica la solución al ejercicio basándote en la siguiente información. Tu explicación debe ser clara y usar formato Markdown y LaTeX.
    
    **PREGUNTA DEL USUARIO:** {query}
    **INFORMACIÓN DE LA SOLUCIÓN:** {context_text}
    
    **TU EXPLICACIÓN:**
    """
    
    try:
        response = model.generate_content(prompt)
        # Devolvemos el texto de la IA y la cadena de Markdown para la imagen
        return response.text, image_markdown
    except Exception as e:
        return f"Error al contactar a la IA: {e}", image_markdown


# -------------------
# INTERFAZ DE USUARIO (SIMPLIFICADA)
# -------------------
df_knowledge = load_knowledge_base()

if df_knowledge is not None:
    st.success("Base de conocimiento cargada. ¡El profesor está listo!")
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¿Qué ejercicio quieres revisar?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # st.markdown mostrará tanto el texto como la imagen

    if prompt := st.chat_input("Escribe el ID del ejercicio (ej: ejercicio 1.1)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Preparando la explicación..."):
                response_text, image_markdown = generate_response(prompt, df_knowledge)
            
            # Combinamos el markdown de la imagen con el texto de la respuesta
            full_response = f"{image_markdown}\n\n{response_text}"
            
            st.markdown(full_response)
            
            # Guardamos la respuesta completa en el historial
            assistant_message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(assistant_message)
else:
    st.error("La aplicación no puede iniciar porque no se pudo cargar la base de conocimiento.")
