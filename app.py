import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="DS Music Matcher", page_icon="🎵")

# --- DATASET (Simulando 10 canciones con características DS) ---
data = {
    'cancion': ['Happy', 'Sad but True', 'Eye of the Tiger', 'Lofi Beats', 'Blinding Lights', 
                'Heavy Metal', 'Chill Jazz', 'Techno Gym', 'Classical Rain', 'Reggaeton Flow'],
    'artista': ['Pharrell', 'Metallica', 'Survivor', 'Lofi Girl', 'The Weeknd', 
                'Slayer', 'Miles Davis', 'DJ Mix', 'Chopin', 'Bad Bunny'],
    'valence': [0.95, 0.10, 0.60, 0.45, 0.85, 0.05, 0.50, 0.70, 0.20, 0.90],
    'energy': [0.80, 0.75, 0.95, 0.15, 0.90, 0.98, 0.25, 0.96, 0.10, 0.85],
}
df = pd.DataFrame(data)

# --- INTERFAZ ---
st.title("🎧 Recomendador en Vivo")
st.info("Ciencia de Datos: Basado en Similitud de Coseno")

with st.sidebar:
    st.header("Tus Datos de Entrada")
    nombre = st.text_input("¿Cómo te llamas?", "Usuario")
    mood = st.slider("Tu Ánimo (0=Triste, 1=Feliz)", 0.0, 1.0, 0.5)
    actividad = st.slider("Tu Energía (0=Dormir, 1=Gym)", 0.0, 1.0, 0.5)

# --- LÓGICA DE RECOMENDACIÓN ---
if st.button("¡Recomiéndame algo!"):
    # 1. Crear el vector del usuario
    user_vector = np.array([[mood, actividad]])
    
    # 2. Obtener los vectores de las canciones
    song_vectors = df[['valence', 'energy']].values
    
    # 3. Calcular Similitud de Coseno
    similitudes = cosine_similarity(user_vector, song_vectors)[0]
    
    # 4. Encontrar la mejor
    indice_top = similitudes.argmax()
    recomendacion = df.iloc[indice_top]
    score = round(similitudes[indice_top] * 100, 2)

    st.success(f"### ¡{nombre}, escucha esto!")
    st.metric(label="Match Algorítmico", value=f"{score}%")
    st.markdown(f"🎵 **Canción:** {recomendacion['cancion']}  \n👤 **Artista:** {recomendacion['artista']}")
    
    # --- GRÁFICO EXPLICATIVO MODIFICADO ---
    st.write("---")
    st.subheader("Explicación de Ciencia de Datos")
    st.write("Mira dónde estás tú vs las canciones en el espacio vectorial:")
    
    # PASO A: Creamos una copia de las canciones y añadimos la etiqueta 'Tipo'
    df_grafico = df.copy()
    df_grafico['Tipo'] = 'Canciones'
    
    # PASO B: Creamos un DataFrame de una sola fila para la posición del usuario
    usuario_df = pd.DataFrame({
        'cancion': ['Tú'],
        'artista': [nombre],
        'valence': [mood],
        'energy': [actividad],
        'Tipo': ['Tu Posición']  # Etiqueta diferente para forzar otro color
    })
    
    # PASO C: Concatenamos ambos DataFrames
    df_final = pd.concat([df_grafico, usuario_df], ignore_index=True)
    
    # PASO D: Graficamos asignando el parámetro color a la columna 'Tipo'
    # Streamlit usará la paleta de colores de su tema (usualmente un color contrastante para ti)
    st.scatter_chart(df_final, x="valence", y="energy", color="Tipo")
    
    st.write(f"Tu posición actual: Valence={mood}, Energy={actividad}")
