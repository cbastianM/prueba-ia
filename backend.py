
# -------------------
# LIBRERÍAS
# -------------------
import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import re

# -------------------
# CONFIGURACIÓN DE LA PÁGINA Y API
# -------------------
st.set_page_config(
    page_title="Profesor Virtual de Ing. Civil",
    page_icon="🏗️",
    layout="wide"
)

# Cargar la API Key desde los secrets de Streamlit
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (FileNotFoundError, KeyError):
    st.error("⚠️ No se encontró la GEMINI_API_KEY. Por favor, créala en .streamlit/secrets.toml")
    st.stop()

# -------------------
# FUNCIONES CON CACHEO
# -------------------

# @st.cache_data: Esta función se ejecutará solo cuando el archivo subido cambie.
@st.cache_data
def load_and_embed_data(uploaded_file):
    """Carga datos desde un archivo Excel y genera los embeddings."""
    if uploaded_file is None:
        return None
    
    try:
        # Lee el archivo excel y especifica que la primera fila es el header
        df = pd.read_excel(uploaded_file, header=0)
        
        # Limpieza de datos
        df.fillna('', inplace=True)
        
        # Generación de embeddings
        model_embedding = 'models/embedding-001'
        
        # Usamos una barra de progreso para la generación de embeddings
        progress_bar = st.progress(0, text="Analizando base de conocimiento...")
        
        def get_embedding(text, index):
            # Actualiza la barra de progreso
            progress_bar.progress(index / len(df), text=f"Procesando entrada {index+1}/{len(df)}...")
            if pd.isna(text) or not str(text).strip():
                return [0.0] * 768
            return genai.embed_content(
                model=model_embedding,
                content=str(text),
                task_type="RETRIEVAL_DOCUMENT"
            )["embedding"]

        df['Embedding'] = [get_embedding(row['Contenido'], i) for i, row in df.iterrows()]
        progress_bar.empty() # Limpia la barra de progreso
        return df

    except Exception as e:
        st.error(f"Error al procesar el archivo Excel: {e}")
        return None

# -------------------
# LÓGICA DEL CHATBOT (CEREBRO CON REGLAS ESTRICTAS)
# -------------------

def extract_exercise_id(query):
    """Extrae un ID de ejercicio como 'BEER 2.73' de la pregunta."""
    books = ["beer", "hibbeler", "singer", "gere", "chopra", "irving"]
    pattern = re.compile(
        r'\b(' + '|'.join(books) + r')[\s\w\.]*(\d+[\.\-]\d+)\b', re.IGNORECASE)
    match = pattern.search(query)
    if match:
        book_name = match.group(1).upper()
        exercise_num = match.group(2).replace('-', '.')
        return f"{book_name} {exercise_num}"
    return None

def generate_response(query, dataframe):
    """Genera una respuesta con el modelo Gemma y lógica estricta."""
    
    # --- CAMBIO DE MODELO A GEMMA ---
    model_generation = genai.GenerativeModel('models/gemma-3-12b-it') # Usamos el modelo Gemma
    
    # --- LÓGICA DE BÚSQUEDA Y RESPUESTA ESTRICTA ---
    exercise_id = extract_exercise_id(query)
    
    # ESTRATEGIA 1: Búsqueda por ID de ejercicio (Prioritaria y Excluyente)
    if exercise_id:
        # Busca una coincidencia exacta y sin distinción de mayúsculas/minúsculas
        match = dataframe[dataframe['ID_Ejercicio'].str.strip().str.upper() == exercise_id.upper()]
        
        if not match.empty:
            # Si se encuentra, el contexto es ÚNICAMENTE la solución
            context = match.iloc[0]['Contenido']
            prompt = f"""
            Tu única tarea es transcribir la siguiente solución. No añadas introducciones, despedidas, o explicaciones adicionales. No digas "La solución es:".
            Simplemente presenta el texto de la solución tal cual se te proporciona.

            **Solución a Transcribir:**
            ---
            {context}
            ---
            """
            response = model_generation.generate_content(prompt)
            return response.text
        else:
            # Si no se encuentra el ID, responde directamente y termina.
            return f"Lo siento, no encuentro el ejercicio '{exercise_id}' en la base de datos que has subido."

    # ESTRATEGIA 2: Búsqueda semántica para preguntas generales (si no se encontró ID)
    query_embedding = genai.embed_content(model='models/embedding-001', content=query, task_type="RETRIEVAL_QUERY")["embedding"]
    
    # Calcular similitud
    dataframe['Embedding'] = dataframe['Embedding'].apply(np.array)
    knowledge_embeddings = np.stack(dataframe['Embedding'].values)
    dot_products = np.dot(knowledge_embeddings, query_embedding)
    top_indices = np.argsort(dot_products)[-1:] # Usamos solo el pasaje más relevante para ser más estrictos
    
    # Comprobar si la similitud es suficiente (umbral)
    similarity_threshold = 0.7 # Puedes ajustar este valor
    if dot_products[top_indices[0]] < similarity_threshold:
        return "Lo siento, no tengo información suficientemente relevante sobre ese tema en mi base de datos."

    context = dataframe.iloc[top_indices]['Contenido'].to_list()[0]
    prompt = f"""
    Basándote ÚNICA Y EXCLUSIVAMENTE en el siguiente texto de contexto, responde la pregunta del usuario.
    Si el contexto no contiene la respuesta, di exactamente: "No tengo información sobre ese tema en mi base de datos."
    No uses conocimiento externo. No infieras. Sé directo.

    **Contexto:**
    ---
    {context}
    ---
    **Pregunta:** {query}
    **Respuesta:**
    """
    response = model_generation.generate_content(prompt)
    return response.text

# -------------------
# INTERFAZ DE USUARIO DE STREAMLIT
# -------------------
st.title("🏗️ Profesor Virtual de Ingeniería Civil")
st.markdown("Sube tu base de conocimiento en formato Excel para comenzar.")

# Cargador de archivos en la barra lateral
with st.sidebar:
    st.header("Configuración")
    uploaded_file = st.file_uploader(
        "Sube tu archivo Excel",
        type=["xlsx"],
        help="El archivo debe tener las columnas: ID_Ejercicio, Libro, Contenido, Tema"
    )

# La lógica principal solo se ejecuta si se ha subido un archivo
if uploaded_file is not None:
    # Carga y procesa los datos. El resultado se guarda en caché.
    # El objeto df_knowledge también se guarda en el estado de la sesión para persistir.
    if 'df_knowledge' not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
        st.session_state.df_knowledge = load_and_embed_data(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name
        # Limpiar historial de chat si se sube un nuevo archivo
        st.session_state.messages = [{"role": "assistant", "content": "¡Base de conocimiento cargada! ¿En qué puedo ayudarte?"}]
        st.rerun()
    
    df_knowledge = st.session_state.df_knowledge

    if df_knowledge is not None:
        st.success(f"¡Archivo '{uploaded_file.name}' cargado y procesado con éxito!")

        # Inicializa y muestra el historial del chat
        if 'messages' not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "¡Base de conocimiento cargada! ¿En qué puedo ayudarte?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Acepta la entrada del usuario
        if prompt := st.chat_input("Pregunta por un ejercicio o un tema..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Genera y muestra la respuesta
            with st.chat_message("assistant"):
                with st.spinner("Consultando la base de conocimiento..."):
                    response = generate_response(prompt, df_knowledge)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Por favor, sube un archivo Excel en la barra lateral para activar el chatbot.")
