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
    model_generation = genai.GenerativeModel('gemma-3-4b-it')
    
    # --- BÚSQUEDA DE CONTEXTO ---
    # ... (esta parte es la misma, no la cambiamos) ...
    extracted_id = extract_exercise_id(query)
    match_row = None
    
    if extracted_id:
        if extracted_id.lower().startswith(('ejercicio', 'problema')):
            match_df = dataframe[dataframe['ID_Ejercicio'].str.strip().str.lower() == extracted_id.lower()]
            if not match_df.empty:
                match_row = match_df.iloc[0]
        else:
            match_df = dataframe[dataframe['Libro'].str.strip().str.upper() == extracted_id.upper()]
            if not match_df.empty:
                match_row = match_df.iloc[0]
    else:
        # ... (lógica de búsqueda semántica) ...
        query_embedding = genai.embed_content(model='models/embedding-001', content=query, task_type="RETRIEVAL_QUERY")["embedding"]
        dataframe['Embedding'] = dataframe['Embedding'].apply(np.array)
        knowledge_embeddings = np.stack(dataframe['Embedding'].values)
        dot_products = np.dot(knowledge_embeddings, query_embedding)
        similarity_threshold = 0.7
        if np.max(dot_products) >= similarity_threshold:
            top_index = np.argmax(dot_products)
            match_row = dataframe.iloc[top_index]

    if match_row is None:
        if extracted_id:
            return f"Lo siento, no he encontrado una entrada para '{extracted_id}' en mi base de conocimiento.", []
        else:
            return "Lo siento, no he encontrado información relevante sobre ese tema en mi base de conocimiento.", []

    # --- DEPURACIÓN PROFUNDA DE IMÁGENES ---
    print("--- INICIO DEPURACIÓN DE IMÁGENES ---")
    
    context_text = match_row['Contenido']
    
    # 1. Comprobamos si la columna 'URL_Imagen' existe y qué contiene
    if 'URL_Imagen' in match_row:
        image_urls_str = match_row['URL_Imagen']
        print(f"DEBUG: Fila encontrada. Contenido de 'URL_Imagen': '{image_urls_str}' (Tipo: {type(image_urls_str)})")
    else:
        print("DEBUG: ERROR - La columna 'URL_Imagen' no se encontró en la fila.")
        image_urls_str = ""

    prompt_parts = []
    images_to_display = []

    # 2. Comprobamos si la cadena de URLs tiene contenido
    if image_urls_str and isinstance(image_urls_str, str) and image_urls_str.strip():
        print("DEBUG: La cadena de URLs no está vacía. Procediendo a dividir y descargar.")
        image_urls = image_urls_str.split('|')
        for i, url in enumerate(image_urls):
            url = url.strip()
            print(f"DEBUG: Intentando descargar imagen {i+1} desde URL: '{url}'")
            try:
                response = requests.get(url)
                # 3. Comprobamos el código de estado de la respuesta
                print(f"DEBUG: Código de estado de la respuesta HTTP: {response.status_code}")
                response.raise_for_status() # Lanza un error para códigos 4xx/5xx
                
                img = Image.open(io.BytesIO(response.content))
                print(f"DEBUG: ¡Imagen {i+1} descargada y procesada correctamente!")
                prompt_parts.append(img)
                images_to_display.append(img)
            except requests.exceptions.RequestException as e:
                print(f"DEBUG: ERROR DE RED al descargar la imagen {i+1}: {e}")
            except Exception as e:
                print(f"DEBUG: ERROR GENERAL al procesar la imagen {i+1}: {e}")
    else:
        print("DEBUG: La cadena de URLs está vacía o no es un string. Saltando la descarga de imágenes.")
        
    print(f"DEBUG: Finalizada la carga de imágenes. Total de imágenes para mostrar: {len(images_to_display)}")
    print("--- FIN DEPURACIÓN DE IMÁGENES ---")

    # --- Lógica del prompt y generación (sin cambios) ---
    prompt_text = f"""
    Eres un profesor de Ingeniería Civil... (tu prompt completo va aquí)
    """
    prompt_parts.insert(0, prompt_text) # Inserta el texto al principio de la lista

    try:
        response_ai = model_generation.generate_content(prompt_parts)
        return response_ai.text, images_to_display
    except Exception as e:
        return f"Ocurrió un error al generar la respuesta: {e}", []

# -------------------
# INTERFAZ DE USUARIO PRINCIPAL (Versión Robusta para Imágenes)
# -------------------
df_knowledge = get_knowledge_base()

# La aplicación solo continúa si la base de conocimiento se cargó correctamente.
if df_knowledge is not None:
    # Este mensaje solo aparece si get_knowledge_base() fue exitoso.
    # st.success("✅ Base de conocimiento lista. Iniciando chat...")

    # Inicializa el historial del chat si no existe.
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu profesor virtual. ¿En qué puedo ayudarte hoy?"}]

    # --- BUCLE 1: Dibuja todo el historial de chat existente ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Comprueba si hay una lista de imágenes en este mensaje del historial.
            # Usamos .get() para evitar errores si la clave "images" no existe.
            if message.get("images"):
                # Si la clave "images" existe y tiene contenido, itera y muestra cada imagen.
                for img in message["images"]:
                    st.image(img, use_column_width=True)
            
            # Siempre muestra el contenido de texto.
            st.markdown(message["content"])

    # --- BUCLE 2: Espera una nueva entrada del usuario ---
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        # Añade el mensaje del usuario al historial y lo muestra en la pantalla.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Genera y muestra la respuesta del asistente.
        with st.chat_message("assistant"):
            with st.spinner("Analizando texto e imágenes..."):
                # Llama a la función que devuelve texto y una LISTA de imágenes.
                response_text, response_images = generate_response(prompt, df_knowledge)
                
                # --- DEBUGGING: Comprueba si se recibieron imágenes ---
                if response_images:
                    st.write(f"DEBUG: Se recibieron {len(response_images)} imágenes para mostrar.")
                    # Muestra las imágenes recibidas inmediatamente.
                    for i, img in enumerate(response_images):
                        st.image(img, caption=f"Referencia {i+1}", use_column_width=True)
                else:
                    st.write("DEBUG: No se recibieron imágenes en la respuesta.")
                
                # Muestra el texto de la respuesta.
                st.markdown(response_text)
                
                # Prepara el mensaje completo (con texto e imágenes) para guardarlo en el historial.
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "images": response_images  # Guarda la lista de imágenes (puede estar vacía).
                }
                st.session_state.messages.append(assistant_message)
else:
    # Este mensaje aparece si get_knowledge_base() devolvió None.
    st.error("No se pudo iniciar el chatbot porque la base de conocimiento no se cargó.")
