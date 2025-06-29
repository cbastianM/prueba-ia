import streamlit as st
import pandas as pd
import google.generativeai as genai
import re

# --- Configuración de la Página de Streamlit (Sin cambios) ---
st.set_page_config(
    page_title="Tutor de Estática con IA",
    page_icon="🏗️",
    layout="wide"
)

# --- Funciones Auxiliares ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('database/statics_problems.csv', dtype={'id': str})
        df['id'] = df['id'].str.strip()
        return df
    except FileNotFoundError:
        st.error("ERROR CRÍTICO: No se encontró 'database/statics_problems.csv'.")
        return None

def find_exercise(query, df):
    # La función de búsqueda robusta que ya teníamos
    normalized_query = query.lower()
    match = re.search(r'(beer|hibbeler)\s*(\d+[\.\-]\d+)', normalized_query)
    if match:
        book, number = match.group(1).strip(), match.group(2).replace('-', '.').strip()
        search_id = f"{book} {number}"
        result_df = df[df['id'] == search_id]
        if not result_df.empty: return result_df.to_dict('records')[0]
    stopwords = ['el','la','de','del','me','puedes','explica','resuelve','problema','ejercicio']
    keywords = [w for w in normalized_query.split() if w not in stopwords and len(w) > 2]
    for keyword in keywords:
        result_df = df[df['tema'].str.contains(keyword, case=False, na=False)]
        if not result_df.empty: return result_df.iloc[0].to_dict()
    for keyword in keywords:
        result_df = df[df['enunciado'].str.contains(keyword, case=False, na=False)]
        if not result_df.empty: return result_df.iloc[0].to_dict()
    return None

# --- FUNCIÓN GET_GEMINI_RESPONSE CON LÓGICA DE FALLBACK ---
def get_gemini_response(api_key, conversation_history, exercise_data):
    """
    Genera una respuesta intentando primero con Gemma 3 y, si falla,
    con Gemini 1.0 Pro como plan B.
    """
    genai.configure(api_key=api_key)
    
    # Lista de modelos a intentar, en orden de preferencia
    models_to_try = [
        'models/gemma-3-12b-it',  # El que queremos (puede que no esté disponible)
        'gemini-1.5-pro-latest', # Una excelente alternativa
        'gemini-1.0-pro'         # El más fiable y compatible
    ]

    # --- El prompt de sistema que ya teníamos, es excelente y no cambia ---
    formatting_rules = """### MANUAL DE ESTILO MATEMÁTICO (OBLIGATORIO) ### ... (etc.)"""
    if exercise_data:
        system_context = f"""Tu rol es ser un tutor experto de Estática... {formatting_rules} ... DATOS DEL PROBLEMA: ..."""
    else:
        system_context = f"Tu rol es ser un tutor general de Estática. {formatting_rules}"
    
    prompt_parts = [system_context, "\n--- HISTORIAL ---"]
    for message in conversation_history:
        prompt_parts.append(f"**{message['role'].replace('user', 'Estudiante').replace('assistant', 'Tutor')}**: {message['content']}")
    full_prompt = "\n".join(prompt_parts)

    # --- Bucle de intento y fallback ---
    for model_name in models_to_try:
        try:
            st.toast(f"Intentando conectar con el modelo: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(full_prompt)
            # Si tiene éxito, informa qué modelo se usó y devuelve la respuesta
            st.toast(f"¡Conexión exitosa! Usando {model_name}.")
            return response.text
        except Exception as e:
            # Si falla, informa del error y prueba el siguiente modelo de la lista
            st.warning(f"No se pudo conectar con '{model_name}'. Intentando el siguiente... Error: {e}")
            continue # Pasa a la siguiente iteración del bucle
    
    # Si todos los modelos fallan, devuelve un error final
    st.error("Error crítico: No se pudo conectar con ninguno de los modelos de IA disponibles. Verifica tu API Key y la configuración del proyecto.")
    return None

# --- RESTO DEL CÓDIGO (SIN CAMBIOS) ---

# Inicialización y Carga de Datos
# ...

# Barra Lateral
with st.sidebar:
    # ... (código de la sidebar sin cambios)
    pass

# Interfaz Principal
st.title("🏗️ Tutor de Estática con IA")
st.markdown("Pide un ejercicio por su ID (ej: `beer 2.43`), por tema, o por parte del enunciado.")

# ... (código del expander de detalles y del historial de chat)
if 'selected_problem' not in st.session_state: st.session_state.selected_problem = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'api_key' not in st.session_state: st.session_state.api_key = None
df_problems = load_data()

with st.sidebar:
    st.header("🔑 Configuración")
    with st.form("api_key_form"):
        api_key_input = st.text_input("Ingresa tu API Key de Google AI", type="password", help="Necesaria para activar el tutor.")
        submitted = st.form_submit_button("Guardar Clave")
        if submitted:
            if api_key_input: st.session_state.api_key = api_key_input; st.success("¡API Key guardada!")
            else: st.warning("El campo está vacío.")
    with st.expander("❓ ¿Cómo obtener una API Key?"):
        st.markdown("1. Ve a [Google AI Studio](https://aistudio.google.com/).\n2. Clic en **'Get API key'** y crea tu clave.")
    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### ✨ Versión Pro")
        st.write("Desbloquea más ejercicios y soporte.")
        st.markdown("[Conoce los beneficios 🚀](https://www.tu-pagina-de-precios.com)")
    st.markdown("---")
    st.header("📚 Ejercicios Disponibles")
    if df_problems is not None:
        for _, row in df_problems.iterrows():
            st.markdown(f"- **ID: {row['id']}** ({row['tema']})")
    else: st.error("No se pudieron cargar los ejercicios.")

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

# Lógica del Chat
if prompt := st.chat_input("¿Qué quieres aprender hoy?"):
    if not st.session_state.api_key:
        st.warning("Por favor, ingresa y guarda tu API Key en la barra lateral."); st.stop()

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    visual_keywords = ["imagen", "pdf", "manual", "visual", "diagrama", "enlace"]
    found_exercise = find_exercise(prompt, df_problems)
    
    if any(keyword in prompt.lower() for keyword in visual_keywords) and st.session_state.selected_problem:
        pass
    elif found_exercise and (st.session_state.selected_problem is None or st.session_state.selected_problem['id'] != found_exercise['id']):
        st.session_state.selected_problem = found_exercise
        with st.chat_message("assistant"):
            with st.spinner("Preparando la explicación inicial... 🤓"):
                initial_history = [{"role": "user", "content": "Explícame cómo resolver este problema paso a paso, usando el formato matemático correcto."}]
                response = get_gemini_response(st.session_state.api_key, initial_history, st.session_state.selected_problem)
                if response:
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    st.error("No se pudo generar la explicación inicial.")
        st.rerun()
    else:
        with st.chat_message("assistant"):
            with st.spinner("Pensando... 🤔"):
                response = get_gemini_response(st.session_state.api_key, st.session_state.chat_history, st.session_state.selected_problem)
            if response:
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                st.error("No se pudo obtener una respuesta.")
    
