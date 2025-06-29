import streamlit as st
import pandas as pd
import google.generativeai as genai
import re

# --- Configuración de la Página de Streamlit (Sin cambios) ---
st.set_page_config(
    page_title="Tutor de Estática con Gemini",
    page_icon="🏗️",
    layout="wide"
)

# --- Funciones Auxiliares ---

@st.cache_data
def load_data():
    """Carga los datos y limpia la columna 'id' para evitar errores de espacios."""
    try:
        # Forzamos que la columna 'id' se lea como texto (string)
        df = pd.read_csv('database/statics_problems.csv', dtype={'id': str})
        # Limpiamos los espacios en blanco de la columna ID, un error muy común.
        df['id'] = df['id'].str.strip()
        return df
    except FileNotFoundError:
        st.error("ERROR CRÍTICO: No se encontró 'database/statics_problems.csv'.")
        return None

# --- FUNCIÓN DE BÚSQUEDA TOTALMENTE NUEVA Y ROBUSTA ---
def find_exercise(query, df):
    """
    Busca un ejercicio usando una estrategia de 3 niveles: ID preciso, tema y enunciado.
    """
    normalized_query = query.lower()

    # --- Nivel 1: Búsqueda por ID preciso (ej. "beer 2.43") ---
    # Regex para buscar patrones como 'libro numero.numero'
    match = re.search(r'(beer|hibbeler)\s*(\d+[\.\-]\d+)', normalized_query)
    if match:
        book = match.group(1).strip()
        number = match.group(2).replace('-', '.').strip()
        search_id = f"{book} {number}"
        
        result_df = df[df['id'] == search_id]
        if not result_df.empty:
            # st.toast(f"Encontrado por ID preciso: {search_id}") # Para depuración
            return result_df.to_dict('records')[0]

    # --- Nivel 2: Búsqueda por palabras clave en la columna "tema" ---
    # Ignoramos palabras comunes para hacer la búsqueda más relevante
    stopwords = ['el', 'la', 'un', 'una', 'de', 'del', 'me', 'puedes', 'explica', 
                 'explícame', 'resuelve', 'problema', 'ejercicio', 'ayuda', 'con']
    keywords = [word for word in normalized_query.split() if word not in stopwords and len(word) > 2]
    
    for keyword in keywords:
        # `na=False` evita errores si hay celdas vacías
        result_df = df[df['tema'].str.contains(keyword, case=False, na=False)]
        if not result_df.empty:
            # st.toast(f"Encontrado por tema con keyword: '{keyword}'") # Para depuración
            return result_df.iloc[0].to_dict() # Devuelve el primer resultado que coincida

    # --- Nivel 3: Búsqueda por palabras clave en la columna "enunciado" ---
    for keyword in keywords:
        result_df = df[df['enunciado'].str.contains(keyword, case=False, na=False)]
        if not result_df.empty:
            # st.toast(f"Encontrado por enunciado con keyword: '{keyword}'") # Para depuración
            return result_df.iloc[0].to_dict()

    return None # Si ningún método funciona, no se encontró nada.


def get_gemini_response(api_key, conversation_history, exercise_data):
    """
    Genera una respuesta de la IA. (Función sin cambios).
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemma-3-12b-it')
        
        # El prompt detallado y estricto que ya teníamos funciona bien aquí
        if exercise_data:
            system_context = f"""
            Tu rol es ser un tutor experto de Estática...
            **REGLAS ESTRICTAS DE FORMATO...**
            **DATOS DEL PROBLEMA:**
            - ID: {exercise_data['id']}
            - Enunciado: {exercise_data['enunciado']}
            - Procedimiento: {exercise_data['procedimiento']}
            - Respuesta: {exercise_data['respuesta']}
            """
        else:
            system_context = "Tu rol es ser un tutor general de Estática..."
        
        prompt_parts = [system_context, "\n--- HISTORIAL ---"]
        for message in conversation_history:
            prompt_parts.append(f"**{message['role'].replace('user', 'Estudiante').replace('assistant', 'Tutor')}**: {message['content']}")
        full_prompt = "\n".join(prompt_parts)
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error al contactar la API de Google Gemini: {e}")
        return None

# --- RESTO DEL CÓDIGO ---

# Inicialización de la memoria de la sesión
if 'selected_problem' not in st.session_state:
    st.session_state.selected_problem = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

df_problems = load_data()

# Interfaz principal y barra lateral (sin cambios)
st.title("🏗️ Tutor de Estática con Google Gemini")
st.markdown("Pide un ejercicio por su ID (ej: `beer 2.43`) o por tema (ej: `lámpara en equilibrio`).")
# ... (código de la sidebar) ...

# Lógica del chat principal (actualizada para usar la nueva función)
if prompt := st.chat_input("¿Qué quieres aprender hoy?"):
    if not st.session_state.api_key:
        st.warning("Por favor, ingresa y guarda tu API Key en la barra lateral."); st.stop()

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    visual_keywords = ["imagen", "pdf", "manual", "visual", "diagrama", "enlace"]
    
    # INTENCIÓN 1: Pedir material visual
    if any(keyword in prompt.lower() for keyword in visual_keywords) and st.session_state.selected_problem:
        # ... (lógica sin cambios)
        pass # La lógica existente aquí es correcta
    
    # INTENCIÓN 2 Y 3: Seleccionar problema o pregunta general
    else:
        # Usamos la nueva y potente función de búsqueda
        found_exercise = find_exercise(prompt, df_problems)
        
        # Comprobamos si el problema encontrado es diferente al que ya está seleccionado
        if found_exercise and (st.session_state.selected_problem is None or st.session_state.selected_problem['id'] != found_exercise['id']):
            st.session_state.selected_problem = found_exercise
            with st.chat_message("assistant"):
                with st.spinner("Preparando la explicación inicial... 🤓"):
                    initial_history = [{"role": "user", "content": "Explícame cómo resolver este problema paso a paso, usando el formato matemático correcto."}]
                    response = get_gemini_response(st.session_state.api_key, initial_history, st.session_state.selected_problem)
                    if response:
                        st.markdown(response)
                        # Creamos un nuevo historial para esta conversación
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    else:
                        st.error("No se pudo generar la explicación inicial.")
            st.rerun()
        
        else: # Si no se encontró un problema nuevo, es una pregunta de seguimiento
            with st.chat_message("assistant"):
                with st.spinner("Pensando... 🤔"):
                    response = get_gemini_response(st.session_state.api_key, st.session_state.chat_history, st.session_state.selected_problem)
                if response:
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    st.error("No se pudo obtener una respuesta.")
