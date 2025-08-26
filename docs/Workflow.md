# Análisis del Flujo de Trabajo del Proyecto

## Introducción

El objetivo de este proyecto es convertir un video largo (ej. una entrevista de YouTube) en un videoclip corto, vertical (9:16), con subtítulos dinámicos y un encuadre profesional, listo para ser publicado en plataformas como Shorts, TikTok o Reels. El script `main.py` actúa como el punto de entrada principal, orquestando el flujo de trabajo a través de `pipeline.py`. El siguiente documento detalla el pipeline técnico paso a paso, desde la URL inicial hasta el video final.

## Diagrama de Flujo

```mermaid
graph TD
    A[URL de YouTube] --> B{1. Descarga};
    B --> C[Video Completo.mp4];
    C --> D{2. Extracción de Audio};
    D --> E[audio.wav];
    E --> F{3. Transcripción y Diarización};
    F --> G((speech.json con timestamps por palabra));
    G --> H{4. Selección de Highlight con LLM};
    H --> I((start_sec, end_sec));
    subgraph "Procesamiento de Video"
        C -- Video --> J{5. Recorte Preciso};
        I -- Timestamps --> J;
        J --> K[cut.mp4];
        K -- Video --> L{6. Cámara Virtual};
        G -- Datos de Hablantes --> L;
        L --> M[cropped.mp4 (silencioso)];
        M -- Video sin audio --> N{7. Muxing Final};
        K -- Audio --> N;
        N --> O[Final.mp4 (sin subtítulos)];
    end
    subgraph "Generación de Subtítulos"
        O -- Video --> P{9. Burn-in de Subtítulos};
        G -- Transcripción --> Q{8. Subtítulos Dinámicos};
        I -- Timestamps --> Q;
        Q --> R[subtitles.ass];
        R -- Archivo .ass --> P;
    end
    P --> S(✅ Final_subtitled.mp4);
```

## Descripción Detallada de Cada Paso

### 1. Descarga (`Components/YoutubeDownloader.py`)

- **Entrada:** Una URL de video (ej. YouTube).
- **Proceso:** Utiliza la librería `pytubefix` para interactuar con YouTube. Presenta al usuario una lista de las calidades de video disponibles para que elija una. Descarga por separado el stream de video seleccionado y el stream de audio de mejor calidad. Finalmente, usa `ffmpeg` para fusionar ambos en un único archivo `mp4`.
- **Salida:** Un archivo de video local con el audio y video de la fuente original.

### 2. Extracción de Audio (`Components/Edit.py`)

- **Entrada:** El video `mp4` descargado.
- **Proceso:** Ejecuta un comando simple de `ffmpeg` para extraer la pista de audio del video y la convierte a un formato estándar (`.wav`, 16kHz, mono), que es el formato óptimo para los modelos de transcripción de Whisper.
- **Salida:** Un archivo `audio.wav`.

### 3. Transcripción y Diarización (`Components/Transcription.py`)

- **Entrada:** El archivo `audio.wav`.
- **Proceso:** Este es un paso clave. Se utiliza el modelo `faster-whisper` para realizar la transcripción. Se activa la opción `word_timestamps=True` para obtener no solo las frases, sino el tiempo de inicio y fin de cada palabra individual. Opcionalmente, si se configura una API key de Hugging Face, utiliza `pyannote.audio` para la diarización (identificar qué hablante dijo cada segmento). La información de los hablantes se fusiona con los segmentos de la transcripción.
- **Salida:** Un archivo `speech.json` muy detallado que contiene una lista de segmentos. Cada segmento tiene el texto, el hablante y una lista de todas sus palabras con sus respectivos timestamps.

### 4. Selección de Highlight (`Components/LanguageTasks.py`)

- **Entrada:** El texto completo de la transcripción.
- **Proceso:** Se envía la transcripción a un Modelo de Lenguaje Grande (LLM) como GPT-4. El prompt le instruye al modelo que actúe como un experto en redes sociales y seleccione el fragmento más interesante, viral o con mayor impacto. 
- **Salida:** Dos valores numéricos: `start_sec` y `end_sec`, que definen la ventana de tiempo del clip a crear.

### 5. Recorte Preciso (`Components/Edit.py`)

- **Entrada:** El video `mp4` original y los timestamps `start_sec` y `end_sec`.
- **Proceso:** Se usa `ffmpeg` para cortar el video. Es crucial que se fuerce el modo de re-codificación (`copy=False`). Esto garantiza que el corte sea exacto a nivel de frame, lo cual es fundamental para mantener la sincronización perfecta con los subtítulos. Si se usara el modo de copia de stream, los cortes solo podrían ocurrir en keyframes, introduciendo un desfase.
- **Salida:** Un video más corto, `cut.mp4`, que contiene el highlight seleccionado.

### 6. Cámara Virtual (`Components/FaceCropYOLO.py`)

- **Entrada:** El video recortado `cut.mp4` y el `speech.json`.
- **Proceso:** Este es el componente más complejo. Redimensiona el video a formato vertical (9:16) y aplica una lógica de "cámara virtual" para mantener al hablante siempre bien encuadrado. 
    - **Modo Estático:** Si hay datos de hablantes, el sistema crea encuadres fijos para cada persona. Cuando hay un cambio de hablante, en lugar de un corte brusco, se realiza una transición suave (paneo y/o zoom) de medio segundo hasta el nuevo encuadre.
    - **Modo Dinámico:** Si no hay datos de hablantes, un estabilizador cinematográfico sigue la cara detectada con un movimiento suave, basado en una simulación de muelle-amortiguador para evitar temblores.
- **Salida:** Un video `cropped.mp4` en formato 9:16, pero sin audio.

### 7. Muxing Final (`Components/FaceCropYOLO.py`)

- **Entrada:** El video `cropped.mp4` (sin audio) y el clip `cut.mp4` (que sí tiene audio).
- **Proceso:** Un comando de `ffmpeg` extrae la pista de audio de `cut.mp4` y la combina con el video de `cropped.mp4`.
- **Salida:** El video `Final.mp4`, que es el clip vertical con audio pero aún sin subtítulos.

### 8. Generación de Subtítulos Dinámicos (`Components/Subtitles.py`)

- **Entrada:** Los datos de transcripción de `speech.json` y los timestamps del highlight.
- **Proceso:** Primero, filtra las palabras que corresponden al clip seleccionado y ajusta sus timestamps para que sean relativos al inicio del nuevo video. Luego, genera un archivo de subtítulos en formato `.ass` (Advanced SubStation Alpha). La lógica agrupa las palabras en líneas cortas basándose en las pausas naturales del habla (un silencio de más de 0.35s inicia una nueva línea), lo que resulta en un ritmo muy natural. Se aplica un estilo de fuente de alto impacto (Impact, tamaño 65) para máxima legibilidad.
- **Salida:** Un archivo `subtitles.ass`.

### 9. Burn-in de Subtítulos (`Components/Subtitles.py`)

- **Entrada:** El video `Final.mp4` y el archivo `subtitles.ass`.
- **Proceso:** El paso final. Se usa `ffmpeg` y su filtro `subtitles` para "quemar" (incrustar) los subtítulos directamente en los frames del video.
- **Salida:** El producto final del pipeline: `Final_subtitled.mp4`.