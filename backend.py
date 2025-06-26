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

# @st.cache_resource: Esta función se ejecuta solo UNA VEZ cuando la app arranca.
# Es perfecto para crear recursos que no cambian, como nuestra base de conocimiento.
@st.cache_resource
def get_knowledge_base():
    """
    Define la base de datos directamente en el código, la convierte a DataFrame
    y genera los embeddings. Retorna el DataFrame procesado.
    """
    # --- ¡AQUÍ ES DONDE AÑADES O MODIFICAS TU CONOCIMIENTO! ---
    data = [
        {
            "ID_Ejercicio": "BEER 2.73",
            "Libro": "Beer & Johnston",
            "Tema": "Estática",
            "Contenido": "La respuesta es la paz."
        },
        {
            "ID_Ejercicio": "HIBBELER 4.5",
            "Libro": "Hibbeler",
            "Tema": "Estructuras",
            "Contenido": "La solución es el amor"
        },
        {
            "ID_Ejercicio": "",
            "Libro": "Teoría",
            "Tema": "Mecánica de Suelos",
            "Contenido": "La consolidación de un suelo es un edificio"
        },
        {
            "ID_Ejercicio": "",
            "Libro": "Teoría",
            "Tema": "Hidráulica",
            "Contenido": "La Ecuación de Manning es un tornillo."
        }
    ]
    
    df = pd.DataFrame(data)
    
    # Generación de embeddings
    model_embedding = 'models/embedding-001'
    def get_embedding(text):
        if pd.isna(text) or not str(text).strip(): return [0.0] * 768
        return genai.embed_content(
            model=model_embedding, content=str(text), task_type="RETRIEVAL_DOCUMENT")["embedding"]

    df['Embedding'] = df['Contenido'].apply(get_embedding)
    return df

# -------------------
# LÓGICA DEL CHATBOT (CEREBRO CON PERSONA DE PROFESOR)
# -------------------

def extract_exercise_id(query):
    """Extrae un ID de ejercicio de la pregunta del usuario."""
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
    """Genera una respuesta con el modelo Gemini y la persona de un profesor que usa Markdown."""
    model_generation = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # ESTRATEGIA 1: Búsqueda por ID de ejercicio (Prioritaria)
    exercise_id = extract_exercise_id(query)
    if exercise_id:
        match = dataframe[dataframe['ID_Ejercicio'].str.strip().str.upper() == exercise_id.upper()]
        if not match.empty:
            context = match.iloc[0]['Contenido']
            libro = match.iloc[0]['Libro']
            
            # --- PROMPT MEJORADO PARA EJERCICIOS ---
            prompt = f"""
            Eres un profesor de Ingeniería Civil. Tu tarea es explicar la solución al ejercicio "{exercise_id}" del libro "{libro}".
            Usa el siguiente texto como base para tu explicación.

            **Instrucciones de formato para tu respuesta:**
            1.  Usa Markdown para estructurar la respuesta (títulos con '#', negritas con '**', listas con '*').
            2.  **MUY IMPORTANTE:** Cualquier fórmula matemática o ecuación debe ser formateada usando LaTeX.
                - Para fórmulas en la misma línea, usa un signo de dólar: $ E = mc^2 $.
                - Para fórmulas en su propia línea, usa dos signos de dólar: $$ V = \frac{{1}}{{n}} R_h^{{2/3}} S^{{1/2}} $$
            3.  Explica los conceptos de forma clara y didáctica, basándote únicamente en el contexto proporcionado.

            **Tu explicación como profesor, usando Markdown y LaTeX:**
            """
            response = model_generation.generate_content(prompt)
            return response.text
        else:
            return f"Lo siento, he buscado en mi base de conocimiento pero no tengo información sobre el ejercicio '{exercise_id}'. Te recomiendo consultar tu libro de texto."

    # ESTRATEGIA 2: Búsqueda semántica para preguntas teóricas
    query_embedding = genai.embed_content(model='models/embedding-001', content=query, task_type="RETRIEVAL_QUERY")["embedding"]
    
    dataframe['Embedding'] = dataframe['Embedding'].apply(np.array)
    knowledge_embeddings = np.stack(dataframe['Embedding'].values)
    dot_products = np.dot(knowledge_embeddings, query_embedding)
    
    similarity_threshold = 0.7 
    if np.max(dot_products) < similarity_threshold:
        return "Lo siento, no he encontrado información suficientemente relevante sobre ese tema en mi base de conocimiento actual."

    top_index = np.argmax(dot_products)
    context = dataframe.iloc[top_index]['Contenido']
    
    # --- PROMPT MEJORADO PARA TEORÍA ---
    prompt = f"""
    Eres un profesor de Ingeniería Civil experto en comunicación visual. Un estudiante te ha hecho la siguiente pregunta.
    Usa ÚNICA Y EXCLUSIVAMENTE el siguiente texto de contexto para formular tu explicación.

    **Instrucciones de formato para tu respuesta:**
    1.  Usa Markdown para estructurar la respuesta (títulos con '#', negritas con '**', listas con '*').
    2.  **MUY IMPORTANTE:** Cualquier fórmula matemática o ecuación debe ser formateada usando LaTeX.
        - Para fórmulas en la misma línea, usa un signo de dólar: $ \sigma = P/A $.
        - Para fórmulas en su propia línea, usa dos signos de dólar: $$ \Sigma F_x = 0 $$
    3.  Explica el concepto de forma clara, concisa y didáctica.

    **Tu explicación como profesor, usando Markdown y LaTeX:**
    """
    response = model_generation.generate_content(prompt)
    return response.text

# -------------------
# INTERFAZ DE USUARIO PRINCIPAL
# -------------------

# Carga la base de conocimiento una sola vez
try:
    df_knowledge = get_knowledge_base()
except Exception as e:
    st.error(f"Ocurrió un error al inicializar la base de conocimiento: {e}")
    st.stop()

# Inicializa el historial del chat
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu profesor virtual. ¿En qué tema o ejercicio necesitas ayuda hoy?"}]

# Muestra los mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Acepta la entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Genera y muestra la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Consultando mis apuntes..."):
            response = generate_response(prompt, df_knowledge)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
