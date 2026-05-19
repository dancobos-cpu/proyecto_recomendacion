import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import altair as alt  # IMPORTANTE: Nueva librería para control total del color

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="DS Music Matcher ULTRA", page_icon="🚀", layout="wide")

# --- DATASET GIGANTE (80 Canciones con Características DS y Géneros) ---
data = {
    'cancion': [
        # 1-10
        'Happy', 'Sad but True', 'Eye of the Tiger', 'Lofi Beats', 'Blinding Lights', 
        'Heavy Metal', 'Chill Jazz', 'Techno Gym', 'Classical Rain', 'Reggaeton Flow',
        # 11-20
        'Pedro Navaja', 'La Rebelión', 'Bad Romance', 'As It Was', 'Resonance',
        'Nightcall', 'Lose Yourself', 'SICKO MODE', 'Do I Wanna Know?', 'Creep',
        # 21-30
        'Dynamite', 'Master of Puppets', 'Chop Suey!', 'Clair de Lune', 'Symphony No. 5',
        'Take Five', 'Blue in Green', 'Three Little Birds', 'Coffee', 'Wake Me Up',
        # 31-40
        'Scary Monsters', 'Bichota', 'Bohemian Rhapsody', 'Stayin Alive', 'Get Lucky',
        'Danza Kuduro', 'La Camisa Negra', 'Bailando', 'Suavemente', 'El Niágara en Bicicleta',
        # 41-50
        'Flowers', 'Bad Guy', 'Stay', 'Umbrella', 'Starboy',
        'In the End', 'Seven Nation Army', 'Sweet Child O\' Mine', 'Viva La Vida', 'Mr. Brightside',
        # 51-60
        'Basket Case', 'The Trooper', 'Psychosocial', 'Killing In The Name', 'God\'s Plan',
        'HUMBLE.', 'Gangsta\'s Paradise', 'Titanium', 'Animals', 'One More Time',
        # 61-70
        'Feeling Good', 'What a Wonderful World', 'Come Fly With Me', 'So What', 'Gymnopédie No. 1',
        'Time', 'The Imperial March', 'Is This Love', 'Riptide', 'Stressed Out',
        # 71-80
        'Gasolina', 'Propuesta Indecente', 'Obsesión', 'La Bilirrubina', 'Take Me Home, Country Roads',
        'Hotel California', 'Smells Like Teen Spirit', 'Wonderwall', 'Another One Bites the Dust', 'Wake Me Up When September Ends'
    ],
    'artista': [
        # 1-10
        'Pharrell', 'Metallica', 'Survivor', 'Lofi Girl', 'The Weeknd', 
        'Slayer', 'Miles Davis', 'DJ Mix', 'Chopin', 'Bad Bunny',
        # 11-20
        'Willie Colón', 'Joe Arroyo', 'Lady Gaga', 'Harry Styles', 'HOME',
        'Kavinsky', 'Eminem', 'Travis Scott', 'Arctic Monkeys', 'Radiohead',
        # 21-30
        'BTS', 'Metallica', 'System of a Down', 'Debussy', 'Beethoven',
        'Dave Brubeck', 'Miles Davis', 'Bob Marley', 'beabadoobee', 'Avicii',
        # 31-40
        'Skrillex', 'Karol G', 'Queen', 'Bee Gees', 'Daft Punk',
        'Don Omar', 'Juanes', 'Enrique Iglesias', 'Elvis Crespo', 'Juan Luis Guerra',
        # 41-50
        'Miley Cyrus', 'Billie Eilish', 'The Kid LAROI', 'Rihanna', 'The Weeknd',
        'Linkin Park', 'The White Stripes', 'Guns N\' Roses', 'Coldplay', 'The Killers',
        # 51-60
        'Green Day', 'Iron Maiden', 'Slipknot', 'Rage Against the Machine', 'Drake',
        'Kendrick Lamar', 'Coolio', 'David Guetta', 'Martin Garrix', 'Daft Punk',
        # 61-70
        'Nina Simone', 'Louis Armstrong', 'Frank Sinatra', 'Miles Davis', 'Erik Satie',
        'Hans Zimmer', 'John Williams', 'Bob Marley', 'Vance Joy', 'Twenty One Pilots',
        # 71-80
        'Daddy Yankee', 'Romeo Santos', 'Aventura', 'Juan Luis Guerra', 'John Denver',
        'Eagles', 'Nirvana', 'Oasis', 'Queen', 'Green Day'
    ],
    'genero': [
        # 1-10
        'Pop', 'Metal', 'Rock', 'Lofi', 'Synthpop', 
        'Metal', 'Jazz', 'Techno', 'Clásica', 'Reggaeton',
        # 11-20
        'Salsa', 'Salsa', 'Pop', 'Pop', 'Synthwave',
        'Synthwave', 'Hip-Hop', 'Hip-Hop', 'Indie', 'Alternative',
        # 21-30
        'K-Pop', 'Metal', 'Metal', 'Clásica', 'Clásica',
        'Jazz', 'Jazz', 'Reggae', 'Lofi', 'EDM',
        # 31-40
        'EDM', 'Reggaeton', 'Rock', 'Disco', 'Funk',
        'Urbano Latino', 'Pop Latino', 'Pop Latino', 'Merengue', 'Bachata/Merengue',
        # 41-50
        'Pop', 'Pop/Alt', 'Pop', 'Pop/R&B', 'R&B/Synth',
        'Nu Metal', 'Rock Alt', 'Hard Rock', 'Pop/Rock', 'Indie Rock',
        # 51-60
        'Punk Rock', 'Heavy Metal', 'Nu Metal', 'Rap Metal', 'Hip-Hop',
        'Hip-Hop', 'Hip-Hop', 'EDM', 'EDM', 'Electronic',
        # 61-70
        'Jazz/Blues', 'Jazz Tradicional', 'Jazz/Vocal', 'Jazz', 'Clásica',
        'Soundtrack', 'Soundtrack', 'Reggae', 'Indie Pop', 'Alternative',
        # 71-80
        'Reggaeton Old School', 'Bachata', 'Bachata', 'Merengue', 'Country',
        'Classic Rock', 'Grunge', 'Britpop', 'Funk/Rock', 'Alternative Rock'
    ],
    'valence': [
        # 1-10
        0.95, 0.10, 0.60, 0.45, 0.85, 0.05, 0.50, 0.70, 0.20, 0.90,
        # 11-20
        0.70, 0.80, 0.65, 0.75, 0.60, 0.35, 0.25, 0.45, 0.40, 0.10,
        # 21-30
        0.90, 0.15, 0.28, 0.35, 0.30, 0.65, 0.15, 0.92, 0.50, 0.75,
        # 31-40
        0.30, 0.85, 0.65, 0.90, 0.80, 0.93, 0.72, 0.88, 0.95, 0.82,
        # 41-50
        0.68, 0.23, 0.70, 0.60, 0.55, 0.21, 0.32, 0.74, 0.56, 0.43,
        # 51-60
        0.82, 0.35, 0.12, 0.45, 0.50, 0.42, 0.51, 0.45, 0.38, 0.86,
        # 61-70
        0.54, 0.88, 0.68, 0.25, 0.25, 0.08, 0.15, 0.82, 0.78, 0.40,
        # 71-80
        0.84, 0.65, 0.58, 0.89, 0.79, 0.41, 0.31, 0.45, 0.76, 0.48
    ],
    'energy': [
        # 1-10
        0.80, 0.75, 0.95, 0.15, 0.90, 0.98, 0.25, 0.96, 0.10, 0.85,
        # 11-20
        0.75, 0.88, 0.88, 0.73, 0.40, 0.55, 0.90, 0.85, 0.65, 0.35,
        # 21-30
        0.85, 0.96, 0.93, 0.05, 0.82, 0.45, 0.10, 0.45, 0.20, 0.90,
        # 31-40
        0.95, 0.80, 0.70, 0.75, 0.80, 0.92, 0.78, 0.83, 0.91, 0.76,
        # 41-50
        0.58, 0.42, 0.78, 0.75, 0.70, 0.89, 0.85, 0.87, 0.45, 0.81,
        # 51-60
        0.91, 0.94, 0.97, 0.95, 0.55, 0.78, 0.68, 0.80, 0.93, 0.78,
        # 61-70
        0.48, 0.20, 0.35, 0.30, 0.02, 0.40, 0.80, 0.52, 0.62, 0.66,
        # 71-80
        0.94, 0.52, 0.48, 0.82, 0.42, 0.50, 0.88, 0.38, 0.72, 0.68
    ],
}

df = pd.DataFrame(data)

# Corrección visual leve de Clair de Lune (Debussy) para gráfico más limpio
# Tenía Energy=0.05 y Valence=0.35, lo subí un poco en el eje X para que no se pegue al borde.
df.loc[df['cancion'] == 'Clair de Lune', 'valence'] = 0.25

# --- INTERFAZ ---
st.title("🎧 Recomendador en Vivo (Versión ULTRA)")
st.info(f"Modelo Espacial de Datos cargado exitosamente con **{len(df)} canciones** distribuidas vectorialmente.")

# Diseño de columnas amplias
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Tus Datos de Entrada")
    nombre = st.text_input("¿Cómo te llamas?", "Melómano")
    mood = st.slider("Tu Ánimo (0=Triste/Melancólico, 1=Feliz/Festivo)", 0.0, 1.0, 0.5)
    actividad = st.slider("Tu Energía (0=Relajado/Dormir, 1=Intenso/Gym)", 0.0, 1.0, 0.5)
    
    ejecutar = st.button("🚀 ¡Ejecutar Similitud de Coseno!")

with col2:
    if ejecutar:
        # 1. Crear vector de usuario
        user_vector = np.array([[mood, actividad]])
        
        # 2. Obtener vectores de canciones
        song_vectors = df[['valence', 'energy']].values
        
        # 3. Calcular Cosenos
        similitudes = cosine_similarity(user_vector, song_vectors)[0]
        
        # 4. Extraer el top match
        indice_top = similitudes.argmax()
        recomendacion = df.iloc[indice_top]
        score = round(similitudes[indice_top] * 100, 2)

        # Despliegue de Resultados
        st.success(f"### ¡{nombre}, el algoritmo ha seleccionado tu pista!")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Precisión del Ángulo Vectorial", value=f"{score}%")
        with metric_col2:
            st.markdown(f"🎵 **Canción:** {recomendacion['cancion']} \n\n👤 **Artista:** {recomendacion['artista']} \n\n🏷️ **Género:** {recomendacion['genero']}")
        
        # --- HISTORIAL LOCAL EN CSV ---
        nueva_interaccion = {
            "Nombre": nombre, "Mood_Usuario": mood, "Energia_Usuario": actividad,
            "Cancion_Recomendada": recomendacion['cancion'], "Artista_Recomendado": recomendacion['artista'],
            "Genero": recomendacion['genero'], "Match_Porcentaje": score
        }
        df_registro = pd.DataFrame([nueva_interaccion])
        archivo_csv = 'historial_usuarios.csv'
        if not os.path.isfile(archivo_csv):
            df_registro.to_csv(archivo_csv, index=False)
        else:
            df_registro.to_csv(archivo_csv, mode='a', header=False, index=False)

        # --- GRÁFICO DE ALTA DENSIDAD MODIFICADO PARA COLOR ROJO ---
        st.write("---")
        st.subheader(f"📍 Mapa de Distribución Vectorial ({len(df)} Tracks)")
        st.write(f"Leyenda: **Punto Rojo** = Tu Posición | Puntos Grises = Canciones del Dataset")

        # Preparación de datos para el gráfico
        df_grafico = df.copy()
        df_grafico['Tipo'] = 'Otras Canciones'
        # Creamos una columna combinada para el tooltip
        df_grafico['Info'] = df_grafico['cancion'] + " - " + df_grafico['artista']
        
        usuario_df = pd.DataFrame({
            'cancion': ['Tú'], 'artista': [nombre], 'genero': ['Usuario'],
            'valence': [mood], 'energy': [actividad], 'Tipo': ['Tu Posición'],
            'Info': [f"Tú ({nombre})"]
        })
        
        df_final = pd.concat([df_grafico, usuario_df], ignore_index=True)

        # --- CREACIÓN DEL GRÁFICO CON ALTAIR ---
        # Esto nos permite mapear explícitamente las categorías a colores específicos.
        # Hacemos las canciones grises y transparentes para que el rojo resalte más.
        
        # Definimos el mapeo de colores
        colores_especificos = alt.Scale(
            domain=['Otras Canciones', 'Tu Posición'],
            range=['#94a3b8', 'red'] # Hexadecimal gris suave y Rojo brillante
        )

        grafico_altair = alt.Chart(df_final).mark_circle(size=100, opacity=0.7).encode(
            x=alt.X('valence', scale=alt.Scale(domain=[0, 1]), title='Ánimo (Valence)'),
            y=alt.Y('energy', scale=alt.Scale(domain=[0, 1]), title='Energía (Energy)'),
            color=alt.Color('Tipo', scale=colores_especificos, legend=None), # Legend=None porque ya lo explicamos en texto arriba
            tooltip=['Info', 'valence', 'energy'] # Qué mostrar al pasar el mouse
        ).interactive() # Permite zoom y paneo

        # Desplegar el gráfico de Altair en Streamlit
        st.altair_chart(grafico_altair, use_container_width=True)
        
        st.caption(f"Ubicación actual en espacio cartesiano -> X: {mood} (Valence) | Y: {actividad} (Energy)")
        st.caption("🔒 Tu interacción ha sido guardada en el historial de forma segura.")
    else:
        st.write("### 👈 Ajusta tus coordenadas emocionales a la izquierda y presiona el botón.")
