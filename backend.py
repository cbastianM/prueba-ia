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
        df['id'] = df['id'].astype(int)
        return df
    except FileNotFoundError:
        st.error("ERROR CRÍTICO: No se encontró 'database/statics_problems.csv'. "
                 "Asegúrate de que el archivo y la carpeta 'database' existan en tu repositorio.")
        return None

def get_gemini_response(api_key, conversation_history, exercise_data):
    """
    Genera una respuesta de la IA usando la API de Google Gemini,
    con instrucciones para formatear las matemáticas y basarse en los datos proporcionados.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemma-3-12b-it')
        
        if exercise_data:
            system_context = f"""
            Tu rol es ser un tutor de Estática que guía al estudiante a través de una solución PREDEFINIDA.
            Tu misión es explicar de forma clara el procedimiento que te proporciono.

            **REGLAS ESTRICTAS DE FORMATO Y CONTENIDO:**
            1.  **FORMATO MATEMÁTICO:** Utiliza SIEMPRE Markdown con sintaxis de LaTeX para todas las ecuaciones y variables.
                -   Para ecuaciones en un bloque separado (display mode), usa dos signos de dólar: `$$ E = mc^2 $$`
                -   Para variables o fórmulas dentro de un texto (inline mode), usa un solo signo de dólar: `La fuerza $F_x$ se calcula así...`
                -   EJEMPLO: Para "Sumatoria de Fx = 0", escribe: `La primera ecuación es $$ \\sum F_x = 0 $$`

            2.  **FUENTE DE VERDAD:** Basa tu explicación ÚNICAMENTE en el procedimiento y la respuesta proporcionados. No inventes pasos, métodos alternativos o información que no esté en los datos.

            3.  **REFERENCIA VISUAL:** Refiérete al material visual (imagen/PDF) que el estudiante ya tiene, diciendo frases como "Observa el diagrama proporcionado...".

            **DATOS DEL PROBLEMA (TU ÚNICA FUENTE DE VERDAD):**
            - **Enunciado:** {exercise_data['enunciado']}
            - **Procedimiento a Explicar:**
              ```
              {exercise_data['procedimiento']}
              ```
            - **Respuesta Final a la que se debe llegar:** `{exercise_data['respuesta']}`

            Ahora, responde la pregunta del estudiante manteniendo estas reglas.
            """
        else:
            system_context = """
            Tu rol es ser un tutor general de Estática. Ayuda con conceptos teóricos o a encontrar un ejercicio. 
            REGLA DE FORMATO: Utiliza SIEMPRE Markdown con sintaxis de LaTeX para todas las ecuaciones y variables.
            - Bloque de ecuación: `$$ ... $$`
            - Ecuación en línea: `$ ... $`
            """
        
        full_prompt = system_context + "\n\n---\n\nHistorial de la conversación:\n" 
        for msg in conversation_history[:-1]:
            full_prompt += f"{msg['role']}: {msg['content']}\n"
        full_prompt += f"Pregunta actual del usuario: {conversation_history[-1]['content']}"

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

# Carga de datos
df_problems = load_data()

# Interfaz principal
st.title("🏗️ Tutor de Estática con Google Gemini")
st.markdown("Pide la solución de un problema por su ID (ej: `resuelve problema 3`) o haz una pregunta general.")

# Barra lateral
with st.sidebar:
    st.header("🔑 Configuración de API")
    api_key_input = st.text_input("Ingresa tu API Key de Google AI Studio", type="password")
    if st.button("Guardar Clave"):
        if api_key_input: st.session_state.api_key = api_key_input; st.success("API Key guardada.")
        else: st.warning("Por favor, ingresa una API Key.")

    with st.expander("❓ ¿Cómo obtener una API Key?"):
        st.markdown("1. Ve a [Google AI Studio](https://aistudio.google.com/).\n2. Haz clic en **'Get API key'**.\n3. Crea y copia tu clave.")
    
    st.header("📚 Ejercicios Disponibles")
    if df_problems is not None:
        for index, row in df_problems.iterrows():
            st.markdown(f"- **ID {row['id']}:** {row['tema']}")
    else:
        st.error("No se pudieron cargar los ejercicios.")

# Mostrar detalles del problema si uno está seleccionado
if st.session_state.selected_problem:
    with st.expander("Detalles del Problema Seleccionado", expanded=True):
        prob = st.session_state.selected_problem
        st.markdown(f"**ID:** {prob['id']} | **Libro:** {prob['libro']} | **Tema:** {prob['tema']}")
        st.markdown(f"**Enunciado:** {prob['enunciado']}")
        if pd.notna(prob['imagen_url']):
            st.markdown(f"**Material Visual:** [**Haga clic aquí para abrir**]({prob['imagen_url']})")

st.markdown("---")

# Mostrar historial de chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Lógica del chat principal
if prompt := st.chat_input("¿Qué quieres aprender hoy?"):
    if not st.session_state.api_key:
        st.warning("Por favor, ingresa y guarda tu API Key de Google en la barra lateral."); st.stop()

    with st.chat_message("user"): st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Lógica de decisión de intenciones
    visual_keywords = ["imagen", "pdf", "manual", "visual", "diagrama", "enlace", "url", "link", "dibujo", "figura", "gráfico"]
    id_match = re.search(r'(id|problema|ejercicio)\s*(\d+)', prompt, re.IGNORECASE)
    
    # INTENCIÓN 1: Pedir material visual
    if any(keyword in prompt.lower() for keyword in visual_keywords) and st.session_state.selected_problem:
        prob = st.session_state.selected_problem
        with st.chat_message("assistant"):
            if pd.notna(prob['imagen_url']):
                assistant_response = f"¡Claro! Aquí tienes el enlace al material visual del problema **ID {prob['id']}**: [**Abrir Imagen/PDF**]({prob['imagen_url']})"
                st.markdown(assistant_response)
            else:
                assistant_response = f"Lo siento, el problema **ID {prob['id']}** no tiene un material visual asociado."
                st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # INTENCIÓN 2: Seleccionar un problema por ID
    elif id_match:
        exercise_id = int(id_match.group(2))
        found_exercise_df = df_problems[df_problems['id'] == exercise_id]
        
        if not found_exercise_df.empty:
            # Si el ID existe, se carga y se explica
            found_exercise = found_exercise_df.to_dict('records')[0]
            st.session_state.selected_problem = found_exercise
            
            with st.chat_message("assistant"):
                with st.spinner("Preparando la explicación inicial... 🤓"):
                    initial_history = [{"role": "user", "content": "Explícame cómo resolver este problema paso a paso, usando el formato matemático correcto."}]
                    response = get_gemini_response(st.session_state.api_key, initial_history, st.session_state.selected_problem)
                    if response:
                        st.markdown(response)
                        # Limpiamos el historial viejo y empezamos uno nuevo con esta conversación
                        st.session_state.chat_history = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}
                        ]
                    else: 
                        st.error("No se pudo generar la explicación inicial.")
            st.rerun()
        else:
            # Si el ID no existe, se informa al usuario
            with st.chat_message("assistant"):
                error_message = f"Lo siento, no pude encontrar el problema con **ID {exercise_id}** en mi base de datos. Por favor, elige un ID válido de la lista en la barra lateral."
                st.markdown(error_message)
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})

    # INTENCIÓN 3: Pregunta de seguimiento o general
    else:
        with st.chat_message("assistant"):
            with st.spinner("Pensando... 🤔"):
                response = get_gemini_response(st.session_state.api_key, st.session_state.chat_history, st.session_state.selected_problem)
            if response:
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else: 
                st.error("No se pudo obtener una respuesta.")
