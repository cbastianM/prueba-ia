# -------------------
# LIBRERÍAS
# -------------------
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import re
from PIL import Image
import io
import requests # Necesitamos requests de nuevo

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
    csv_url = "https://raw.githubusercontent.com/cbastianM/prueba-ia/main/Conocimiento_Ing_Civil.csv"
    try:
        # skipinitialspace=True ayuda a limpiar espacios después de las comas
        df = pd.read_csv(csv_url, skipinitialspace=True)
        # Reemplaza valores nulos (NaN) por strings vacíos
        df.fillna('', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ Error al leer el archivo CSV desde GitHub: {e}")
        return None

# -------------------
# LÓGICA DEL CHATBOT
# -------------------
def generate_response(query, dataframe):
    """Genera una respuesta, buscando el ejercicio y descargando la imagen si existe."""
    
    # Limpieza de la búsqueda
    query_cleaned = query.strip().lower()
    
    # Búsqueda robusta
    match_df = dataframe[dataframe['ID_Ejercicio'].str.strip().str.lower() == query_cleaned]
    
    if match_df.empty:
        return f"Lo siento, no pude encontrar el ejercicio '{query}'. Por favor, asegúrate de escribir el identificador exacto (ej: 'ejercicio 1.1').", []

    match_row = match_df.iloc[0]

    # --- PROCESAMIENTO DE IMÁGENES (CON LIMPIEZA DE URL) ---
    context_text = match_row['Contenido']
    image_url = match_row['URL_Imagen']
    images_to_display = []
    
    if image_url and isinstance(image_url, str):
        # --- ¡SOLUCIÓN CLAVE! ---
        # Limpiamos la URL de espacios y cualquier comilla que pueda tener
        cleaned_url = image_url.strip().strip("'\"")
        
        try:
            response = requests.get(cleaned_url)
            response.raise_for_status() # Lanza un error si la descarga falla
            img = Image.open(io.BytesIO(response.content))
            images_to_display.append(img)
        except Exception as e:
            # Si falla la descarga, añadimos un aviso al texto de la respuesta
            context_text += f"\n\n**(Aviso del sistema: No se pudo cargar la imagen asociada. Error: {e})**"
            
    # --- GENERACIÓN DE LA RESPUESTA DE LA IA ---
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt_parts = [
        f"""
        **ROL:** Eres un profesor de Ingeniería Civil.
        **TAREA:** Explica la solución al ejercicio basándote en la información y la imagen proporcionadas.
        **PREGUNTA DEL USUARIO:** {query}
        **INFORMACIÓN DE LA SOLUCIÓN:** {context_text}
        
        **INSTRUCCIONES:**
        1.  Describe la solución encontrada en 'INFORMACIÓN DE LA SOLUCIÓN'.
        2.  Si hay una imagen, analízala y úsala para enriquecer tu explicación. Di "Como se ve en el diagrama..." o algo similar.
        3.  Formatea tu respuesta con Markdown y ecuaciones en LaTeX.
        
        **TU EXPLICACIÓN:**
        """
    ]
    
    # Añadimos las imágenes (si las hay) al prompt para la IA
    if images_to_display:
        prompt_parts.extend(images_to_display)

    try:
        response = model.generate_content(prompt_parts)
        # Devolvemos el texto de la IA y las imágenes que ya descargamos para mostrar
        return response.text, images_to_display
    except Exception as e:
        return f"Error al contactar a la IA: {e}", images_to_display

# -------------------
# INTERFAZ DE USUARIO
# -------------------
df_knowledge = load_knowledge_base()

if df_knowledge is not None:
    st.success("Base de conocimiento cargada. ¡El profesor está listo!")
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¿Qué ejercicio quieres revisar?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("images"):
                for img in message["images"]:
                    st.image(img, use_column_width=True)
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe el ID del ejercicio (ej: ejercicio 1.1)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Preparando la explicación..."):
                response_text, response_images = generate_response(prompt, df_knowledge)
            
            if response_images:
                for img in response_images:
                    st.image(img, use_column_width=True)
            
            st.markdown(response_text)
            
            assistant_message = {"role": "assistant", "content": response_text, "images": response_images}
            st.session_state.messages.append(assistant_message)
else:
    st.error("La aplicación no puede iniciar porque no se pudo cargar la base de conocimiento.")
