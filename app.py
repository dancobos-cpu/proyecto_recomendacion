import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="DS Music Matcher", page_icon="🎵")

# --- DATASET (Simulando 15 canciones con características DS) ---
# Valence: 0.0 (triste) a 1.0 (feliz) | Energy: 0.0 (calma) a 1.0 (entrenar)
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
    nombre = st.text_input("¿Cómo te llamas?")
    mood = st.slider("Tu Ánimo (0=Triste, 1=Feliz)", 0.0, 1.0, 0.5)
    actividad = st.slider("Tu Energía (0=Dormir, 1=Gym)", 0.0, 1.0, 0.5)

# --- LÓGICA DE RECOMENDACIÓN ---
if st.button("¡Recomiéndame algo!"):
    # 1. Crear el vector del usuario
    user_vector = np.array([[mood, actividad]])
    
    # 2. Obtener los vectores de las canciones
    song_vectors = df[['valence', 'energy']].values
    
    # 3. Calcular Similitud de Coseno
    # Esto mide el ángulo entre el usuario y las canciones en un plano 2D
    similitudes = cosine_similarity(user_vector, song_vectors)[0]
    
    # 4. Encontrar la mejor
    indice_top = similitudes.argmax()
    recomendacion = df.iloc[indice_top]
    score = round(similitudes[indice_top] * 100, 2)

    st.success(f"### ¡{nombre}, escucha esto!")
    st.metric(label="Match Algorítmico", value=f"{score}%")
    st.markdown(f"🎵 **Canción:** {recomendacion['cancion']}  \n👤 **Artista:** {recomendacion['artista']}")
    
    # --- GRÁFICO EXPLICATIVO ---
    st.write("---")
    st.subheader("Explicación de Ciencia de Datos")
    st.write("Mira dónde estás tú (punto rojo) vs las canciones en el espacio vectorial:")
    st.scatter_chart(df, x="valence", y="energy", color="#94a3b8")
    st.write(f"Tu posición: Valence={mood}, Energy={actividad}")