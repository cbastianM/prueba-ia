import streamlit as st
import pandas as pd

st.set_page_config(page_title="Diagnóstico Final", layout="wide")
st.title("🕵️‍♂️ Diagnóstico Final de Lectura de Datos")

# --- FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data(ttl=60) # Cache muy corto para asegurar que siempre recargue los cambios
def load_data():
    """Lee el CSV desde una URL fija y lo devuelve."""
    csv_url = "https://raw.githubusercontent.com/cbastianM/prueba-ia/main/base_conocimiento.csv"
    try:
        df = pd.read_csv(csv_url)
        # Reemplaza valores nulos (NaN) por strings vacíos
        df.fillna('', inplace=True)
        return df
    except Exception as e:
        st.error(f"❌ ERROR AL LEER EL CSV DESDE GITHUB: {e}")
        return None

# --- EJECUCIÓN PRINCIPAL ---
df = load_data()

if df is not None:
    st.success("✅ Archivo CSV cargado en un DataFrame de Pandas.")
    
    st.header("1. Contenido del DataFrame Cargado")
    st.markdown("Verifica que las columnas y los datos se vean correctos aquí.")
    st.dataframe(df)
    
    st.header("2. Prueba de Búsqueda Manual")
    st.markdown("Escribe un ID de la tabla de arriba (ej: `ejercicio 1.1`) y presiona 'Buscar'.")
    
    # Input para que el usuario escriba el ID
    search_id = st.text_input("ID del Ejercicio a buscar:")
    
    if st.button("Buscar"):
        if search_id:
            # LIMPIEZA: Quitamos espacios y convertimos a minúsculas en ambos lados
            query_cleaned = search_id.strip().lower()
            df['ID_Ejercicio_cleaned'] = df['ID_Ejercicio'].str.strip().str.lower()
            
            # Búsqueda
            match_df = df[df['ID_Ejercicio_cleaned'] == query_cleaned]
            
            st.divider()
            if not match_df.empty:
                st.success(f"✅ ¡MATCH ENCONTRADO PARA '{search_id}'!")
                
                # Selecciona la primera fila encontrada
                match_row = match_df.iloc[0]
                
                st.subheader("Datos de la Fila Encontrada:")
                st.write(match_row)
                
                st.subheader("Intento de mostrar la imagen:")
                image_data = match_row['URL_Imagen']
                
                if image_data and isinstance(image_data, str) and image_data.strip():
                    st.write("Se encontró contenido en la columna 'URL_Imagen'. Intentando mostrarla:")
                    try:
                        st.image(image_data, caption="Imagen cargada desde la Base de Datos")
                        st.success("¡La imagen se mostró correctamente!")
                    except Exception as e:
                        st.error(f"FALLO al intentar mostrar la imagen. El dato es: '{image_data}'. Error: {e}")
                else:
                    st.warning("La columna 'URL_Imagen' para esta fila está vacía.")
                    
            else:
                st.error(f"❌ NO SE ENCONTRÓ MATCH PARA '{search_id}'.")
                st.warning("Posibles causas: El ID no existe en la tabla o fue escrito incorrectamente.")
                st.write("IDs disponibles (en minúsculas):", df['ID_Ejercicio_cleaned'].tolist())
        else:
            st.warning("Por favor, escribe un ID para buscar.")
else:
    st.error("La aplicación no puede continuar porque el DataFrame no se cargó.")
