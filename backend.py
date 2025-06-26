import streamlit as st
import pandas as pd

st.set_page_config(page_title="Diagnóstico Final", layout="wide")
st.title("🕵️‍♂️ Diagnóstico Final de Lectura de Datos")

# --- FUNCIÓN DE CARGA DE DATOS ---
@st.cache_data(ttl=60)
def load_data():
    csv_url = "https://raw.githubusercontent.com/cbastianM/prueba-ia/main/Conocimiento_Ing_Civil.csv"
    try:
        # Añadimos 'skipinitialspace=True' para que pandas limpie los espacios al leer
        df = pd.read_csv(csv_url, skipinitialspace=True)
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
    st.dataframe(df)
    
    st.header("2. Prueba de Búsqueda Manual")
    search_id = st.text_input("ID del Ejercicio a buscar:")
    
    if st.button("Buscar"):
        if search_id:
            # --- LÓGICA DE BÚSQUEDA MEJORADA ---
            # Limpiamos la entrada del usuario de la misma forma
            query_cleaned = search_id.strip().lower()
            
            # Aplicamos la misma limpieza a la columna del DataFrame para la comparación
            # .str.strip() quita espacios al inicio y al final
            # .str.lower() convierte a minúsculas
            match_df = df[df['ID_Ejercicio'].str.strip().str.lower() == query_cleaned]
            # --- FIN DE LA LÓGICA MEJORADA ---

            st.divider()
            if not match_df.empty:
                st.success(f"✅ ¡MATCH ENCONTRADO PARA '{search_id}'!")
                match_row = match_df.iloc[0]
                
                st.subheader("Datos de la Fila Encontrada:")
                st.write(match_row)
                
                st.subheader("Intento de mostrar la imagen:")
                image_data = match_row['URL_Imagen']
                
                if image_data and isinstance(image_data, str) and image_data.strip():
                    st.write("Se encontró contenido en la columna 'URL_Imagen'. Intentando mostrarla:")
                    try:
                        st.image(image_data, caption="Imagen cargada")
                        st.success("¡La imagen se mostró correctamente!")
                    except Exception as e:
                        st.error(f"FALLO al intentar mostrar la imagen. Error: {e}")
                else:
                    st.warning("La columna 'URL_Imagen' para esta fila está vacía.")
            else:
                st.error(f"❌ NO SE ENCONTRÓ MATCH PARA '{search_id}'.")
                st.warning("Verifica que el ID exista en la tabla de arriba.")
                # Mostramos los valores limpios de la columna para ver si hay algún problema
                st.write("Valores limpios disponibles en 'ID_Ejercicio':", df['ID_Ejercicio'].str.strip().str.lower().tolist())
        else:
            st.warning("Por favor, escribe un ID para buscar.")
else:
    st.error("La aplicación no puede continuar porque el DataFrame no se cargó.")
