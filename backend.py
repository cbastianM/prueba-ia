# -------------------
# LIBRERÍAS (ya no necesitamos requests ni io para la imagen)
# -------------------
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import re
from PIL import Image
import base64 # Librería estándar para decodificar

# -------------------
# CONFIGURACIÓN DE LA PÁGINA Y API
# -------------------
st.set_page_config(page_title="Profesor de Ing. Civil", page_icon="🎓")
st.title("🎓 Profesor Virtual de Ingeniería Civil")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.error("⚠️ No se encontró la GEMINI_API_KEY. Configúrala en los secrets de Streamlit.")
    st.stop()

# -------------------
# CARGA DE DATOS DESDE CSV EN GITHUB
# -------------------
@st.cache_data(ttl=3600)
def load_knowledge_base():
    """Lee el CSV desde una URL fija de GitHub."""
    csv_url = "https://raw.githubusercontent.com/cbastianM/prueba-ia/main/base_conocimiento.csv"
    try:
        df = pd.read_csv(csv_url)
        df.fillna('', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Error al leer el archivo CSV desde GitHub: {e}")
        return None

# -------------------
# LÓGICA DEL CHATBOT CON BASE64
# -------------------
def generate_response(query, dataframe):
    """Genera una respuesta, buscando el ejercicio y usando la imagen Base64 si existe."""
    query_lower = query.lower().strip()
    match_row = None
    
    match_df = dataframe.loc[dataframe['ID_Ejercicio'].str.lower() == query_lower]
    
    if not match_df.empty:
        match_row = match_df.iloc[0]

    if match_row is None:
        return "Lo siento, no pude encontrar ese ejercicio. Por favor, escribe el ID exacto (ej: 'ejercicio 1.1').", []

    context_text = match_row['Contenido']
    base64_string = match_row['URL_Imagen'] # Ahora es una cadena base64
    images_to_display = []
    
    # Prepara el prompt para la IA
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt_parts = [f"**Pregunta del estudiante:** {query}\n\n**Información del ejercicio:** {context_text}\n\n**Tu explicación como profesor:**"]

    # Si hay una cadena base64, la preparamos para la IA y para mostrarla
    if base64_string and base64_string.startswith('data:image'):
        # Creamos el objeto de imagen para la IA
        image_part = {
            "mime_type": "image/png", # Asumimos PNG, puedes cambiarlo
            "data": base64_string.split(",")[1] # Quitamos el prefijo "data:image/png;base64,"
        }
        prompt_parts.append(image_part)
        # La cadena base64 se puede pasar directamente a st.image
        images_to_display.append(base64_string)

    try:
        response = model.generate_content(prompt_parts)
        return response.text, images_to_display
    except Exception as e:
        return f"Error al contactar a la IA: {e}", images_to_display


# -------------------
# INTERFAZ DE USUARIO
# -------------------
df_knowledge = load_knowledge_base()

if df_knowledge is not None:
    st.success("Base de conocimiento cargada.")
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¿Qué ejercicio quieres revisar?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("images"):
                for img_data in message["images"]:
                    st.image(img_data, use_column_width=True) # st.image puede manejar cadenas base64
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe el ID del ejercicio (ej: ejercicio 1.1)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_text, response_images = generate_response(prompt, df_knowledge)
            
            if response_images:
                for img_data in response_images:
                    st.image(img_data, use_column_width=True)
            
            st.markdown(response_text)
            
            assistant_message = {"role": "assistant", "content": response_text, "images": response_images}
            st.session_state.messages.append(assistant_message)
