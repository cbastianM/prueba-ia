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
    Genera una respuesta con un modelo de IA con dos personalidades:
    1. ESTRICTO para ejercicios: Solo usa la base de datos.
    2. FLEXIBLE para teoría: Usa su conocimiento general, enriquecido opcionalmente por la base de datos.
    """
    model_generation = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # --- PASO 1: Determinar si la pregunta es sobre un ejercicio específico ---
    exercise_id = extract_exercise_id(query)
    
    # --- CAMINO A: PREGUNTA SOBRE UN EJERCICIO (PERSONALIDAD ESTRICTA) ---
    if exercise_id:
        print(f"DEBUG: ID de ejercicio detectado: '{exercise_id}'. Entrando en CAMINO ESTRICTO.") # Mensaje de depuración
        
        match = dataframe[dataframe['ID_Ejercicio'].str.strip().str.upper() == exercise_id.upper()]
        
        if not match.empty:
            print("DEBUG: Ejercicio encontrado en la base de datos.") # Mensaje de depuración
            context = match.iloc[0]['Contenido']
            libro = match.iloc[0]['Libro']
            
            prompt_ejercicio = f"""
            **ROL Y OBJETIVO:** Eres un profesor asistente cuya única tarea es explicar la solución del ejercicio "{exercise_id}" del libro "{libro}".

            **REGLAS CRÍTICAS:**
            1.  Tu conocimiento se limita ESTRICTA Y ÚNICAMENTE al siguiente "Contexto de la Solución".
            2.  NO PUEDES usar conocimiento externo ni inventar información.
            3.  Tu explicación debe ser didáctica. Formatea con Markdown y usa LaTeX para las ecuaciones (ej: $ ... $ o $$ ... $$).

            **Contexto de la Solución (Tu única fuente de verdad):**
            ---
            {context}
            ---

            **Explicación:**
            """
            response = model_generation.generate_content(prompt_ejercicio)
            return response.text
        else:
            print("DEBUG: Ejercicio NO encontrado en la base de datos.") # Mensaje de depuración
            return f"He revisado mis apuntes y no tengo la solución para el ejercicio '{exercise_id}'. Para problemas específicos, solo puedo usar la información registrada en mi base de datos."

    # --- CAMINO B: PREGUNTA TEÓRICA O GENERAL (PERSONALIDAD FLEXIBLE) ---
    else:
        print(f"DEBUG: No se detectó ID de ejercicio. Entrando en CAMINO FLEXIBLE para la pregunta: '{query}'") # Mensaje de depuración
        
        # En este camino, no necesitamos buscar en nuestra base de datos, ya que el modelo usará su conocimiento general.
        # Simplemente le pasamos la pregunta directamente con un prompt que le da libertad.
        
        prompt_teoria = f"""
        **ROL Y OBJETIVO:** Eres un profesor de Ingeniería Civil experto, amigable y apasionado. Tu objetivo es responder a la pregunta de un estudiante de la forma más completa y clara posible.

        **REGLAS CRÍTICAS:**
        1.  **Usa tu conocimiento general:** Tienes total libertad para usar todo tu conocimiento como modelo de IA avanzado para responder a la pregunta.
        2.  **Sé un gran profesor:** Explica los conceptos de forma intuitiva, da ejemplos si es necesario y estructura tu respuesta para que sea fácil de seguir.
        3.  **Formato Impecable:** Utiliza Markdown (negritas, listas, etc.) y formatea cualquier ecuación, fórmula o variable matemática con LaTeX (ej: $ ... $ o $$ ... $$).

        **Pregunta del Estudiante:**
        ---
        {query}
        ---

        **Tu respuesta como profesor experto:**
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
