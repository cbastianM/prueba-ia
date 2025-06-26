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
    """
    Extrae un ID de ejercicio de la pregunta del usuario.
    Puede reconocer dos formatos:
    1. Nombre del libro + número (ej: "Beer 2.43")
    2. Palabra "ejercicio" + número (ej: "ejercicio 1.1")
    """
    query_lower = query.lower()

    # FORMATO 1: Buscar "ejercicio" o "problema" seguido de un número
    exercise_pattern = re.compile(r'\b(ejercicio|problema)\s+(\d+\.\d+)\b', re.IGNORECASE)
    exercise_match = exercise_pattern.search(query_lower)
    if exercise_match:
        # Reconstruimos el ID para que coincida con la base de datos, ej: "ejercicio 1.1"
        return f"{exercise_match.group(1)} {exercise_match.group(2)}"

    # FORMATO 2: Buscar nombre del libro + número (como respaldo)
    books = ["beer", "hibbeler", "singer", "gere", "chopra", "irving"]
    book_pattern = re.compile(
        r'\b(' + '|'.join(books) + r')[\s\w\.]*(\d+[\.\-]\d+)\b', re.IGNORECASE)
    book_match = book_pattern.search(query_lower)
    if book_match:
        # Devolvemos el formato "LIBRO NUMERO" que se buscará en la columna 'Libro'
        book_name = book_match.group(1).upper()
        exercise_num = book_match.group(2).replace('-', '.')
        return f"{book_name} {exercise_num}"
        
    return None

def generate_response(query, dataframe):
    """
    Genera una respuesta multimodal.
    Busca el ID en la columna 'ID_Ejercicio' o 'Libro' según el formato.
    """
    model_generation = genai.GenerativeModel('gemma-3-27b-it')
    
    # --- BÚSQUEDA DE CONTEXTO ---
    extracted_id = extract_exercise_id(query)
    match_row = None
    
    if extracted_id:
        # Comprobamos si el ID extraído empieza con "ejercicio" o "problema"
        if extracted_id.lower().startswith(('ejercicio', 'problema')):
            # --- CAMINO A: Búsqueda por ID de Ejercicio ---
            print(f"DEBUG: Buscando en columna 'ID_Ejercicio' por '{extracted_id}'")
            match_df = dataframe[dataframe['ID_Ejercicio'].str.strip().str.lower() == extracted_id.lower()]
            if not match_df.empty:
                match_row = match_df.iloc[0]
        else:
            # --- CAMINO B: Búsqueda por Nombre de Libro ---
            print(f"DEBUG: Buscando en columna 'Libro' por '{extracted_id}'")
            match_df = dataframe[dataframe['Libro'].str.strip().str.upper() == extracted_id.upper()]
            if not match_df.empty:
                match_row = match_df.iloc[0]
    else:
        # --- CAMINO C: Búsqueda Semántica (si no hay ID) ---
        print(f"DEBUG: No se encontró ID. Realizando búsqueda semántica.")
        query_embedding = genai.embed_content(model='models/embedding-001', content=query, task_type="RETRIEVAL_QUERY")["embedding"]
        dataframe['Embedding'] = dataframe['Embedding'].apply(np.array)
        knowledge_embeddings = np.stack(dataframe['Embedding'].values)
        dot_products = np.dot(knowledge_embeddings, query_embedding)
        
        similarity_threshold = 0.7
        if np.max(dot_products) >= similarity_threshold:
            top_index = np.argmax(dot_products)
            match_row = dataframe.iloc[top_index]

    # Si no se encontró ningún contenido relevante, devuelve un mensaje
    if match_row is None:
        if extracted_id:
            return f"Lo siento, no he encontrado una entrada para '{extracted_id}' en mi base de conocimiento.", []
        else:
            return "Lo siento, no he encontrado información relevante sobre ese tema en mi base de conocimiento.", []

    # --- LÓGICA MULTIMODAL (esta parte no necesita cambios) ---
    context_text = match_row['Contenido']
    image_urls_str = match_row['URL_Imagen']
    
    prompt_parts = []
    images_to_display = []

    prompt_text = f"""
    Eres un profesor de Ingeniería Civil. Analiza el siguiente texto y, si se proporcionan imágenes, úsalas para explicar la pregunta del estudiante.

    **Contexto:**
    {context_text}

    **Pregunta:**
    {query}

    **Instrucciones Clave:**
    1.  Si hay imágenes, tu explicación DEBE hacer referencia a ellas. Describe lo que muestran y cómo se relacionan con el tema. Usa frases como "En la primera imagen vemos...", "La segunda imagen ilustra...", etc.
    2.  Si no hay imágenes, responde basándote solo en el texto.
    3.  Explica los conceptos con claridad, formato Markdown y ecuaciones en LaTeX ($...$$).

    **Tu Explicación:**
    """
    prompt_parts.append(prompt_text)

    if image_urls_str and isinstance(image_urls_str, str) and image_urls_str.strip():
        image_urls = image_urls_str.split('|')
        for url in image_urls:
            try:
                response = requests.get(url.strip())
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                prompt_parts.append(img)
                images_to_display.append(img)
            except Exception as e:
                print(f"Error al descargar la imagen: {e}")

    try:
        response_ai = model_generation.generate_content(prompt_parts)
        return response_ai.text, images_to_display
    except Exception as e:
        return f"Ocurrió un error al generar la respuesta: {e}", []
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
