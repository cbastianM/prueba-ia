# -------------------
# LIBRERÍAS
# -------------------
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import re
import requests
from PIL import Image
import io
# -------------------
# CONFIGURACIÓN DE LA PÁGINA Y API
# -------------------
st.set_page_config(
    page_title="Tu Profesor de Ing. Civil",
    page_icon="🎓",
    layout="centered" # Un layout más enfocado para chat
)

st.title("🎓 Tu Profesor Virtual de Ingeniería Civil")
st.markdown("Hazme una pregunta sobre un tema o pide la explicación de un ejercicio de la base de conocimiento.")

# Cargar la API Key desde los secrets de Streamlit
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (FileNotFoundError, KeyError):
    st.error("⚠️ No se encontró la GEMINI_API_KEY. Por favor, configúrala en los secrets de Streamlit Cloud.")
    st.stop()

# -------------------
# BASE DE CONOCIMIENTO (DATOS DENTRO DEL CÓDIGO)
# -------------------

# Reemplaza tu antigua función get_knowledge_base() por esta:

@st.cache_resource
def get_knowledge_base():
    """
    Lee un archivo CSV desde una URL pública (GitHub), lo convierte a DataFrame
    y genera los embeddings. Retorna el DataFrame procesado.
    """
    # --- CONSTRUYE LA URL DEL ARCHIVO CSV EN GITHUB ---
    # Reemplaza 'tu_usuario_github', 'tu_repositorio' y 'main' si es necesario.
    github_user = "cbastianM"
    github_repo = "prueba-ia"
    branch_name = "main" # O 'master', dependiendo de tu repositorio
    file_path = "Conocimiento_Ing_Civil.csv"

    csv_url = f"https://raw.githubusercontent.com/cbastianM/prueba-ia/refs/heads/main/Conocimiento_Ing_Civil.csv"

    try:
        # Lee el archivo CSV directamente desde la URL
        df = pd.read_csv(csv_url)
        
        # Limpieza de datos: reemplaza valores nulos (NaN) por strings vacíos
        df.fillna('', inplace=True)
        
        # Generación de embeddings
        model_embedding = 'models/embedding-001'
        
        # Usamos una barra de progreso para dar feedback al usuario durante la carga inicial
        progress_bar = st.progress(0, text="Analizando base de conocimiento...")
        
        def get_embedding(text, index):
            # Actualiza la barra de progreso
            progress_bar.progress((index + 1) / len(df), text=f"Procesando entrada {index+1}/{len(df)}...")
            if not isinstance(text, str) or not text.strip():
                return [0.0] * 768
            return genai.embed_content(
                model=model_embedding, content=text, task_type="RETRIEVAL_DOCUMENT")["embedding"]

        # Aplica la función de embedding
        df['Embedding'] = [get_embedding(row['Contenido'], i) for i, row in df.iterrows()]
        
        # Limpia la barra de progreso una vez terminado
        progress_bar.empty()
        
        return df

    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo CSV desde GitHub: {e}")
        st.warning("Verifica que la URL sea correcta y que el archivo CSV esté en el repositorio.")
        return None


# -------------------
# LÓGICA DEL CHATBOT MULTIMODAL
# -------------------
def extract_exercise_id(query):
    """Extrae un ID de ejercicio como 'BEER 2.73' de la pregunta."""
    books = ["beer", "hibbeler", "singer", "gere", "chopra", "irving"]
    pattern = re.compile(
        r'\b(' + '|'.join(books) + r')[\s\w\.]*(\d+[\.\-]\d+)\b', re.IGNORECASE)
    match = pattern.search(query)
    if match:
        book_name, exercise_num = match.group(1).upper(), match.group(2).replace('-', '.')
        return f"{book_name} {exercise_num}"
    return None

# -------------------
# INTERFAZ DE USUARIO PRINCIPAL (Modificada para mostrar múltiples imágenes)
# -------------------
df_knowledge = get_knowledge_base()

if df_knowledge is not None:
    st.success("✅ Base de conocimiento cargada. ¡El profesor está listo!")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte hoy?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "images" in message:  # Busca la clave "images" (plural)
                for i, img in enumerate(message["images"]): # Itera sobre la lista de imágenes
                    st.image(img, caption=f"Imagen {i+1}", use_column_width=True) # Añadido use_column_width
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizando texto e imágenes..."):
                response_text, response_images = generate_response(prompt, df_knowledge) # Recibe la lista de imágenes
                
                # Prepara el mensaje para el historial (con la lista de imágenes)
                assistant_message = {"role": "assistant", "content": response_text, "images": response_images}
                
                # Muestra cada imagen con su caption
                if response_images:
                    for i, img in enumerate(response_images):
                        st.image(img, caption=f"Imagen {i+1} de referencia", use_column_width=True) # Añadido use_column_width
                
                st.markdown(response_text)
                st.session_state.messages.append(assistant_message)
else:
    st.error("La base de conocimiento no pudo ser cargada. Revisa la URL del CSV y los logs.")
