import streamlit as st
import pandas as pd
import google.generativeai as genai
import re

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Tutor de Estática con Gemini",
    page_icon="🏗️",
    layout="wide"
)

# --- Funciones Auxiliares ---

@st.cache_data
def load_data():
    """Carga los datos de problemas desde el archivo CSV."""
    try:
        df = pd.read_csv('database/statics_problems.csv')
        # ¡IMPORTANTE! Ya no convertimos el ID a entero. Se queda como texto.
        # df['id'] = df['id'].astype(int) <-- LÍNEA ELIMINADA
        return df
    except FileNotFoundError:
        st.error("ERROR CRÍTICO: No se encontró 'database/statics_problems.csv'.")
        return None

# --- FUNCIÓN DE BÚSQUEDA COMPLETAMENTE REESCRITA ---
def find_exercise_by_string_id(query, df):
    """
    Busca un ejercicio usando un patrón de texto como "beer 2.43".
    Es flexible con espacios y separadores (puntos o guiones).
    """
    # Patrón de Regex: busca (beer|hibbeler), seguido de espacios, seguido de (número.número) o (número-número)
    # Puedes añadir más libros a la lista, ej: (beer|hibbeler|meriam|...)
    match = re.search(r'(beer|hibbeler)\s*(\d+[\.\-]\d+)', query, re.IGNORECASE)
    
    if match:
        book_name = match.group(1).lower()
        problem_number = match.group(2).replace('-', '.') # Normalizamos guiones a puntos
        
        # Construimos el ID estandarizado para buscar en el DataFrame
        search_id = f"{book_name} {problem_number}"
        
        # Buscamos el ID exacto en el DataFrame
        result_df = df[df['id'] == search_id]
        
        if not result_df.empty:
            return result_df.to_dict('records')[0] # Devuelve el diccionario del problema encontrado
    
    return None # Si no hay match o no se encuentra el ID, devuelve None


def get_gemini_response(api_key, conversation_history, exercise_data):
    """
    Genera una respuesta de la IA. (Esta función no necesita cambios).
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemma-3-12b-it')
        
        if exercise_data:
            # ... (prompt largo y detallado que ya teníamos)
            system_context = f"""
            Tu rol es ser un tutor de Estática...
            **REGLAS ESTRICTAS...**
            **DATOS DEL PROBLEMA:**
            - ID: {exercise_data['id']}
            - Enunciado: {exercise_data['enunciado']}
            - Procedimiento: {exercise_data['procedimiento']}
            - Respuesta: {exercise_data['respuesta']}
            """
        else:
            system_context = """
            Tu rol es ser un tutor general de Estática...
            """
        
        full_prompt = system_context + "\n\n---\n\nHistorial: " + "".join([f"{m['role']}: {m['content']}\n" for m in conversation_history])
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error al contactar la API de Google Gemini: {e}")
        return None

# --- INICIO DE LA APLICACIÓN ---

# Inicialización y carga de datos
# ... (código sin cambios)
if 'selected_problem' not in st.session_state:
    st.session_state.selected_problem = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
df_problems = load_data()

# Interfaz principal
st.title("🏗️ Tutor de Estática con Google Gemini")
st.markdown("Pide un ejercicio por su nombre y número (ej: `explica beer 2.43`)")

# Barra lateral
with st.sidebar:
    st.header("🔑 Configuración de API")
    # ... (código de API key y Versión Pro sin cambios)
    with st.container(border=True):
        st.markdown("#### ✨ Versión Pro")
        st.write("Desbloquea ejercicios ilimitados, explicaciones avanzadas y soporte prioritario.")
        st.link_button("Conoce los beneficios 🚀", "https://www.tu-pagina-de-precios.com")
    st.markdown("---")
    st.header("📚 Ejercicios Disponibles")
    if df_problems is not None:
        for index, row in df_problems.iterrows():
            # Ahora muestra el ID de texto correctamente
            st.markdown(f"- **ID: {row['id']}** ({row['tema']})")
    else:
        st.error("No se pudieron cargar los ejercicios.")

# Mostrar detalles del problema seleccionado
# ... (código sin cambios)
if st.session_state.selected_problem:
    with st.expander("Detalles del Problema Seleccionado", expanded=True):
        st.markdown("---")
        # ... (muestra detalles)


# Mostrar historial de chat
# ... (código sin cambios)
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- LÓGICA DE CHAT ACTUALIZADA ---
if prompt := st.chat_input("¿Qué quieres aprender hoy?"):
    if not st.session_state.api_key:
        st.warning("Por favor, ingresa y guarda tu API Key de Google en la barra lateral."); st.stop()

    with st.chat_message("user"): st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    visual_keywords = ["imagen", "pdf", "manual", "visual", "diagrama", "enlace", "url", "link", "dibujo", "figura", "gráfico"]
    
    # INTENCIÓN 1: Pedir material visual
    if any(keyword in prompt.lower() for keyword in visual_keywords) and st.session_state.selected_problem:
        # ... (código sin cambios)
        prob = st.session_state.selected_problem
        with st.chat_message("assistant"):
            if pd.notna(prob['imagen_url']):
                assistant_response = f"¡Claro! Aquí tienes el enlace al material visual del problema **{prob['id']}**: [**Abrir Imagen/PDF**]({prob['imagen_url']})"
                st.markdown(assistant_response)
            else:
                assistant_response = f"Lo siento, el problema **{prob['id']}** no tiene un material visual asociado."
                st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # INTENCIÓN 2: Seleccionar un problema por su nuevo ID de texto
    else:
        found_exercise = find_exercise_by_string_id(prompt, df_problems)
        
        if found_exercise:
            # Si el ID existe, se carga y se explica
            st.session_state.selected_problem = found_exercise
            
            with st.chat_message("assistant"):
                with st.spinner("Preparando la explicación inicial... 🤓"):
                    initial_history = [{"role": "user", "content": "Explícame cómo resolver este problema paso a paso, usando el formato matemático correcto."}]
                    response = get_gemini_response(st.session_state.api_key, initial_history, st.session_state.selected_problem)
                    if response:
                        st.markdown(response)
                        st.session_state.chat_history = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
                    else: 
                        st.error("No se pudo generar la explicación inicial.")
            st.rerun()
        else:
            # INTENCIÓN 3: No se encontró un ID, es una pregunta de seguimiento o general
            with st.chat_message("assistant"):
                with st.spinner("Pensando... 🤔"):
                    response = get_gemini_response(st.session_state.api_key, st.session_state.chat_history, st.session_state.selected_problem)
                if response:
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else: 
                    st.error("No se pudo obtener una respuesta.")
