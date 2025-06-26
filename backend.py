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

     # La URL para acceder al archivo en formato "raw" (crudo)
    csv_url = f"https://raw.githubusercontent.com/{github_user}/{github_repo}/{branch_name}/{file_path}"
    
    st.info(f"Cargando base de conocimiento desde: {csv_url}")


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
# LÓGICA DEL CHATBOT (CEREBRO CON PERSONA DE PROFESOR)
# -------------------

# --- FUNCIÓN AUXILIAR (ASEGÚRATE DE QUE ESTÉ AQUÍ) ---
def extract_exercise_id(query):
    """Extrae un ID de ejercicio de la pregunta del usuario."""
    # Puedes añadir más nombres de libros a esta lista
    books = ["beer", "hibbeler", "singer", "gere", "chopra", "irving"]
    
    # Patrón de expresión regular para encontrar "libro numero.numero"
    pattern = re.compile(
        r'\b(' + '|'.join(books) + r')'  # Busca una de las palabras de la lista de libros
        r'[\s\w\.]*'                      # Permite texto intermedio
        r'(\d+[\.\-]\d+)\b',              # Captura el número del ejercicio como "2.73" o "4-5"
        re.IGNORECASE                     # Ignora mayúsculas/minúsculas
    )
    
    match = pattern.search(query)
    if match:
        book_name = match.group(1).upper()
        exercise_num = match.group(2).replace('-', '.')
        return f"{book_name} {exercise_num}"
    return None

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
# INTERFAZ DE USUARIO PRINCIPAL (VERSIÓN ROBUSTA)
# -------------------

# Paso 1: Intentar cargar la base de conocimiento.
# Usamos un bloque try-except para manejar cualquier posible error durante la carga.
try:
    # Esta línea llama a la función que tiene los datos y los procesa.
    # El resultado se guarda en caché para no repetirlo.
    df_knowledge = get_knowledge_base() 
    
    # Comprobación explícita para asegurarnos de que el DataFrame no está vacío o es inválido.
    if df_knowledge is None or df_knowledge.empty:
        st.error("❌ La base de conocimiento no se pudo cargar o está vacía. El chatbot no puede funcionar.")
        # Detenemos la ejecución aquí si no hay datos.
        st.stop() 

except Exception as e:
    st.error(f"🚨 Ocurrió un error crítico al inicializar la base de conocimiento: {e}")
    st.warning("Revisa la función 'get_knowledge_base' en el código fuente.")
    # Detenemos la ejecución si hay un error.
    st.stop()


# Si el código llega hasta aquí, significa que df_knowledge se cargó correctamente.
# Ahora podemos dibujar la interfaz de chat con seguridad.

st.success("✅ Base de conocimiento cargada. ¡El profesor está listo!")

# Inicializa el historial del chat si no existe.
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy tu profesor virtual. ¿En qué tema o ejercicio necesitas ayuda hoy?"}]

# Muestra todos los mensajes del historial en cada recarga de la página.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LA BARRA DE CHAT ---
# Esta línea dibuja la barra de entrada de texto en la parte inferior.
# El `if` se ejecuta solo cuando el usuario escribe algo y presiona Enter.
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # 1. Añade el mensaje del usuario al historial y lo muestra en la pantalla.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Genera y muestra la respuesta del asistente.
    with st.chat_message("assistant"):
        # Muestra un indicador de "pensando" mientras se genera la respuesta.
        with st.spinner("Consultando mis apuntes y formulando una respuesta..."):
            response = generate_response(prompt, df_knowledge)
            st.markdown(response)
    
    # 3. Añade la respuesta del asistente al historial para que persista.
    st.session_state.messages.append({"role": "assistant", "content": response})
