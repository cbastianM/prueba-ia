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

def generate_response(query, dataframe):
    """
    Genera una respuesta usando un modelo de IA con dos personalidades:
    1.  ESTRICTO para ejercicios: Solo usa la base de datos.
    2.  FLEXIBLE para teoría: Puede usar su conocimiento general.
    """
    model_generation = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # --- PASO 1: Determinar si la pregunta es sobre un ejercicio específico ---
    exercise_id = extract_exercise_id(query)
    
    # --- CAMINO A: PREGUNTA SOBRE UN EJERCICIO (PERSONALIDAD ESTRICTA) ---
    if exercise_id:
        match = dataframe[dataframe['ID_Ejercicio'].str.strip().str.upper() == exercise_id.upper()]
        
        if not match.empty:
            # Si el ejercicio ESTÁ en la base de datos
            context = match.iloc[0]['Contenido']
            libro = match.iloc[0]['Libro']
            
            prompt_ejercicio = f"""
            Eres un profesor asistente. Tu única tarea es explicar la solución al ejercicio "{exercise_id}" del libro "{libro}".
            
            **Reglas Estrictas:**
            1.  Basa tu explicación ÚNICA Y EXCLUSIVAMENTE en el siguiente "Contexto de la Solución".
            2.  No puedes añadir información, datos o métodos que no estén en el contexto.
            3.  Explica los pasos de forma clara, didáctica y usando formato Markdown y LaTeX para las ecuaciones.
            
            **Contexto de la Solución (Tu única fuente de verdad):**
            ---
            {context}
            ---

            **Tu explicación como profesor asistente:**
            """
            response = model_generation.generate_content(prompt_ejercicio)
            return response.text
        else:
            # Si el ejercicio NO ESTÁ en la base de datos
            return f"Lo siento, he buscado en mis apuntes pero no tengo registrada la solución para el ejercicio '{exercise_id}'. Para los ejercicios, solo puedo proporcionar la información que tengo en mi base de datos."

    # --- CAMINO B: PREGUNTA TEÓRICA O GENERAL (PERSONALIDAD FLEXIBLE) ---
    else:
        # Para preguntas generales, intentamos enriquecer la respuesta con la base de datos,
        # pero permitimos que el modelo use su conocimiento general.
        
        query_embedding = genai.embed_content(model='models/embedding-001', content=query, task_type="RETRIEVAL_QUERY")["embedding"]
        
        dataframe['Embedding'] = dataframe['Embedding'].apply(np.array)
        knowledge_embeddings = np.stack(dataframe['Embedding'].values)
        dot_products = np.dot(knowledge_embeddings, query_embedding)
        
        # Encontramos el contexto más relevante, si existe.
        context = ""
        similarity_threshold = 0.7 
        if np.max(dot_products) >= similarity_threshold:
            top_index = np.argmax(dot_products)
            context = dataframe.iloc[top_index]['Contenido']

        prompt_teoria = f"""
        Eres un profesor de Ingeniería Civil amable, experto y apasionado.
        Un estudiante te ha hecho una pregunta. Debes responderla de la mejor manera posible.

        **Tus Guías de Conversación:**
        1.  **Usa tu conocimiento general:** Eres libre de usar todo tu conocimiento como modelo de IA para responder a la pregunta de forma completa y detallada.
        2.  **Si hay contexto, úsalo:** Abajo te proporciono "Información Relevante de mis Apuntes". Si este texto es útil para responder la pregunta, intégralo en tu explicación, quizás diciendo "Esto me recuerda a un apunte que tengo..." o "Para complementar, mis notas dicen que...". Si no es relevante, puedes ignorarlo.
        3.  **Formato Impecable:** Usa Markdown para que tu respuesta sea clara (títulos, negritas, listas). Formatea todas las ecuaciones y variables matemáticas con LaTeX ($...$ o $$...$$).
        
        **Información Relevante de mis Apuntes (Opcional):**
        ---
        {context}
        ---

        **Pregunta del Estudiante:**
        {query}

        **Tu Respuesta como Profesor Experto:**
        """
        response = model_generation.generate_content(prompt_teoria)
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
