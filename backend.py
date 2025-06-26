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
    """
    Genera una respuesta interactiva y didáctica, usando la base de datos como
    única fuente de conocimiento, pero con la libertad de explicar y razonar.
    """
    model_generation = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # La búsqueda de contexto sigue un proceso similar, pero el prompt cambia todo.
    # Primero, buscamos el contexto más relevante.
    
    # Paso 1: Intentar encontrar un ID de ejercicio.
    exercise_id = extract_exercise_id(query)
    if exercise_id:
        match = dataframe[dataframe['ID_Ejercicio'].str.strip().str.upper() == exercise_id.upper()]
        if match.empty:
            # Si se menciona un ID pero no se encuentra, paramos aquí.
            return f"Lo siento, he buscado en mis apuntes pero no tengo información sobre el ejercicio '{exercise_id}'. ¿Puedo ayudarte con otro tema?"
        else:
            # Si se encuentra, usamos ese contexto específico.
            context = match.iloc[0]['Contenido']
            libro = match.iloc[0]['Libro']
            tema = match.iloc[0]['Tema']
            context_source = f"el ejercicio {exercise_id} del libro {libro} sobre {tema}"
    else:
        # Paso 2: Si no hay ID, hacer una búsqueda semántica para encontrar los fragmentos más relevantes.
        query_embedding = genai.embed_content(model='models/embedding-001', content=query, task_type="RETRIEVAL_QUERY")["embedding"]
        
        dataframe['Embedding'] = dataframe['Embedding'].apply(np.array)
        knowledge_embeddings = np.stack(dataframe['Embedding'].values)
        dot_products = np.dot(knowledge_embeddings, query_embedding)
        
        # Obtenemos los 2 fragmentos más relevantes para dar más riqueza al contexto.
        top_indices = np.argsort(dot_products)[-2:][::-1]
        
        # Umbral de similitud para asegurar relevancia.
        similarity_threshold = 0.7
        if np.max(dot_products) < similarity_threshold:
            return "Lo siento, no he encontrado información suficientemente relevante sobre ese tema en mi base de conocimiento actual para darte una respuesta confiable."

        # Unimos los fragmentos de contexto encontrados.
        relevant_passages = dataframe.iloc[top_indices]['Contenido'].tolist()
        context = "\n\n---\n\n".join(relevant_passages)
        context_source = "mis apuntes de la base de conocimiento"

    # --- EL NUEVO PROMPT INTELIGENTE ---
    # Este prompt le da al modelo el poder de ser un verdadero profesor.
    prompt = f"""
    Eres un profesor de Ingeniería Civil amable, paciente y muy didáctico.
    Tu misión es responder a la pregunta de un estudiante.

    **Tus Reglas de Oro:**
    1.  **Fuente de Verdad Única:** Debes basar tu respuesta ÚNICA Y EXCLUSIVAMENTE en la siguiente "Información de Contexto". No puedes usar conocimiento externo ni inventar datos. Tu conocimiento se limita a lo que está escrito abajo.
    2.  **Sé un Profesor, no un Loro:** No te limites a copiar el texto. Debes explicarlo, simplificarlo, compararlo o usarlo para responder directamente a la pregunta específica del usuario. Adáptate al tono y la intención de la pregunta.
    3.  **Formato Profesional:** Usa Markdown para que tu respuesta sea clara y fácil de leer. Utiliza negritas, listas y, sobre todo, formatea las ecuaciones con LaTeX ($...$ para inline, $$...$$ para bloque).
    4.  **Si no puedes, dilo:** Si la pregunta del estudiante requiere información que no está en el contexto (por ejemplo, comparar con un tema ausente o realizar un cálculo para el cual no tienes los datos), debes indicarlo amablemente diciendo algo como: "Esa es una excelente pregunta, pero mi conocimiento actual sobre este tema se basa solo en {context_source} y no incluye esa información específica."

    ---
    **Información de Contexto (Tu única fuente de verdad):**
    {context}
    ---

    **Pregunta del Estudiante:**
    {query}

    **Tu Respuesta como Profesor:**
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
