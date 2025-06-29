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

# --- Funciones Auxiliares (Sin cambios) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('database/statics_problems.csv')
        return df
    except FileNotFoundError:
        st.error("ERROR CRÍTICO: No se encontró 'database/statics_problems.csv'.")
        return None

def find_exercise_by_string_id(query, df):
    match = re.search(r'(beer|hibbeler)\s*(\d+[\.\-]\d+)', query, re.IGNORECASE)
    if match:
        book_name = match.group(1).lower()
        problem_number = match.group(2).replace('-', '.')
        search_id = f"{book_name} {problem_number}"
        result_df = df[df['id'] == search_id]
        if not result_df.empty:
            return result_df.to_dict('records')[0]
    return None

def get_gemini_response(api_key, conversation_history, exercise_data):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.0-pro')
        
        if exercise_data:
            system_context = f"""
            Tu rol es ser un tutor de Estática... (El prompt largo y detallado que ya teníamos)
            **REGLAS ESTRICTAS:**
            1.  **FORMATO MATEMÁTICO:** Usa `$$...$$` para bloques y `$..$` en línea.
            2.  **FUENTE DE VERDAD:** Basa tu explicación ÚNICAMENTE en el procedimiento y respuesta proporcionados.
            3.  **REFERENCIA VISUAL:** Refiérete al material visual que el estudiante ya tiene.
            **DATOS DEL PROBLEMA:**
            - ID: {exercise_data['id']}
            - Enunciado: {exercise_data['enunciado']}
            - Procedimiento: {exercise_data['procedimiento']}
            - Respuesta: {exercise_data['respuesta']}
            """
        else:
            system_context = "Tu rol es ser un tutor general de Estática..."
        
        prompt_parts = [system_context, "\n--- HISTORIAL DE CONVERSACIÓN ---"]
        for message in conversation_history:
            prompt_parts.append(f"**{message['role'].replace('user', 'Estudiante').replace('assistant', 'Tutor')}**: {message['content']}")
        
        full_prompt = "\n".join(prompt_parts)
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error al contactar la API de Google Gemini: {e}")
        return None

# --- INICIO DE LA APLICACIÓN ---

# Inicialización de la memoria de la sesión
if 'selected_problem' not in st.session_state:
    st.session_state.selected_problem = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

df_problems = load_data()

# --- INICIO DE LA SECCIÓN CORREGIDA: BARRA LATERAL ---
with st.sidebar:
    st.header("🔑 Configuración")
    
    # Usamos un formulario para agrupar los widgets de la API Key.
    # Esto asegura que se envíen juntos y previene reruns no deseados.
    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "Ingresa tu API Key de Google AI", 
            type="password",
            help="Necesaria para activar el tutor de IA."
        )
        submitted = st.form_submit_button("Guardar Clave")
        
        if submitted:
            if api_key_input:
                st.session_state.api_key = api_key_input
                st.success("¡API Key guardada correctamente!")
            else:
                st.warning("El campo de la API Key está vacío.")

    with st.expander("❓ ¿Cómo obtener una API Key?"):
        st.markdown("""
        1.  Ve a [Google AI Studio](https://aistudio.google.com/).
        2.  Inicia sesión y haz clic en **'Get API key'**.
        3.  Crea una nueva clave en tu proyecto y cópiala.
        """)

    st.markdown("---")

    # Separamos la sección "Pro" en su propio contenedor
    with st.container(border=True):
        st.markdown("#### ✨ Versión Pro")
        st.write("Desbloquea más ejercicios y soporte prioritario.")
        # Usamos markdown para el enlace, que es más estable que link_button
        st.markdown("[Conoce los beneficios 🚀](https://www.tu-pagina-de-precios.com)") # Reemplaza con tu URL

    st.markdown("---")

    st.header("📚 Ejercicios Disponibles")
    if df_problems is not None:
        for index, row in df_problems.iterrows():
            st.markdown(f"- **ID: {row['id']}** ({row['tema']})")
    else:
        st.error("No se pudieron cargar los ejercicios.")
# --- FIN DE LA SECCIÓN CORREGIDA: BARRA LATERAL ---

# Interfaz principal
st.title("🏗️ Tutor de Estática con Google Gemini")
st.markdown("Pide un ejercicio por su nombre y número (ej: `explica beer 2.43`)")

if st.session_state.selected_problem:
    with st.expander("Detalles del Problema Seleccionado", expanded=True):
        prob = st.session_state.selected_problem
        st.markdown(f"**ID:** {prob['id']} | **Libro:** {prob['libro']}")
        st.markdown(f"**Enunciado:** {prob['enunciado']}")
        if pd.notna(prob['imagen_url']):
            st.markdown(f"**Material Visual:** [**Haga clic aquí para abrir**]({prob['imagen_url']})")

st.markdown("---")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Lógica de Chat (sin cambios)
if prompt := st.chat_input("¿Qué quieres aprender hoy?"):
    if not st.session_state.api_key:
        st.warning("Por favor, ingresa y guarda tu API Key en la barra lateral."); st.stop()

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    visual_keywords = ["imagen", "pdf", "manual", "visual", "diagrama", "enlace", "url", "dibujo", "figura"]
    found_exercise = find_exercise_by_string_id(prompt, df_problems)
    
    if any(keyword in prompt.lower() for keyword in visual_keywords) and st.session_state.selected_problem:
        prob = st.session_state.selected_problem
        with st.chat_message("assistant"):
            if pd.notna(prob['imagen_url']):
                assistant_response = f"¡Claro! Aquí tienes el enlace al material visual del problema **{prob['id']}**: [**Abrir Imagen/PDF**]({prob['imagen_url']})"
            else:
                assistant_response = f"Lo siento, el problema **{prob['id']}** no tiene un material visual asociado."
            st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    
    elif found_exercise:
        st.session_state.selected_problem = found_exercise
        with st.chat_message("assistant"):
            with st.spinner("Preparando la explicación inicial... 🤓"):
                initial_history = [{"role": "user", "content": "Explícame cómo resolver este problema paso a paso, usando el formato matemático correcto."}]
                response = get_gemini_response(st.session_state.api_key, initial_history, st.session_state.selected_problem)
                if response:
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else: st.error("No se pudo generar la explicación inicial.")
        st.rerun()
    
    else:
        with st.chat_message("assistant"):
            with st.spinner("Pensando... 🤔"):
                response = get_gemini_response(st.session_state.api_key, st.session_state.chat_history, st.session_state.selected_problem)
            if response:
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else: st.error("No se pudo obtener una respuesta.")
