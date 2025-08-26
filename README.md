# AI Shorts Generator 🎬🤖

> Generador automático de **YouTube Shorts** a partir de videos largos usando Whisper, OpenAI GPT y OpenCV/FFmpeg. Convierte entrevistas o podcasts en clips atractivos en formato vertical listos para subir a TikTok, Reels y Shorts.

---

## ✨ Características Destacadas (v0.1)

- 📥 **Descarga de Video:** Soporte para YouTube y otras fuentes.
- 🎙️ **Transcripción Precisa:** Transcripción automática con `faster-whisper` y timestamps a nivel de palabra.
- 🧠 **Selección Inteligente:** Uso de **GPT-4** para identificar segmentos "virales".
- 🗣️ **Diarización de Hablantes:** Detección de quién habla y cuándo.
- 🎥 **Cámara Virtual Inteligente:** Pan/zoom dinámico o encuadres fijos por hablante con transiciones suaves.
- 🔥 **Subtítulos Dinámicos:** Generación automática en `.ass` con sincronización por pausas y estilo viral.
- 🐳 **Soporte Docker:** Para despliegue en cualquier entorno.

---

## 🚀 Primeros Pasos

### 📦 Requisitos

- **Python 3.8+**
- **FFmpeg** instalado y accesible desde la terminal (`ffmpeg -version`)
- **Cuenta de OpenAI** y API Key activa

### ⚙️ Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/ahernandez-developer/InterviewIAShorts.git
cd InterviewIAShorts

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API Key de OpenAI
# (Crea un archivo .env a partir de .env.example y añade tu API key)
```

### ▶️ Uso

El script es interactivo. Ejecútalo desde la raíz del proyecto:

```bash
python main.py
```

Luego, introduce la URL del video de YouTube cuando se te solicite.

---

## 📂 Estructura de Salida

Cada video procesado genera una carpeta en el directorio `out/` con la siguiente estructura:

```
out/
└── <nombre_del_video>/
    ├── Final.mp4               # Video vertical sin subtítulos
    ├── Final_subtitled.mp4     # ✅ Video final con subtítulos incrustados
    └── subtitles.ass           # Archivo de subtítulos dinámicos
```
Adicionalmente, la carpeta `work/` contiene archivos intermedios como el audio, la transcripción completa (`speech.json`), etc., que se mantienen para facilitar el debugging y el seguimiento del flujo de trabajo.

---

## 📊 Roadmap

El desarrollo del "Generador de Shorts Virales con IA" sigue un enfoque iterativo, priorizando la estabilidad y la calidad del resultado final.

### Logros de la v0.1 (Completado)

- [x] Soporte para `faster-whisper` para transcripciones rápidas y precisas.
- [x] Generación de subtítulos dinámicos en formato `.ass`.
- [x] Agrupación de subtítulos por pausas naturales del habla.
- [x] Cámara estática con transiciones suaves entre hablantes.
- [x] Lógica de recorte de video precisa para evitar desincronización.

### Próximos Pasos (Priorizado)

#### Fase 1.5: Calidad de Edición Profesional (Prioridad Actual)

*El objetivo de esta fase es asegurar que cada clip generado cumpla con un estándar de calidad profesional, con una duración consistente y una edición visual de alto impacto.*

- [x] **Consistencia en la Duración del Clip:**
    - [x] Modificar el prompt y la lógica en `LanguageTasks.py` para instruir al LLM a seleccionar un *highlight* que dure estrictamente **entre 50 y 70 segundos**.
- [x] **Edición y Seguimiento de Cámara Profesional:**
    - [x] **Refinamiento del Seguimiento Facial:** Se implementaron mejoras significativas en `FaceCropYOLO.py` para asegurar un seguimiento de caras más suave y natural. Esto incluyó:
        -   **Corrección de la Relación de Aspecto:** Eliminación de estiramientos visuales al garantizar que la ventana de recorte siempre mantenga una relación de aspecto 9:16.
        -   **Seguimiento Robusto del Hablante:** Mejora del centrado de la cámara y reducción de momentos en los que apunta a "nada", mediante la alineación de los timestamps de `speech.json` con el video recortado y una lógica de búsqueda de turnos más tolerante.
    - [ ] **Transiciones Cinematográficas:** Mejorar las transiciones entre hablantes. Investigar y aplicar curvas de animación más sofisticadas (e.g., *ease-in-out*) y paneos/zooms más sutiles para un acabado profesional.

#### Fase 1: Refinamiento y Estabilidad (v0.2) - ¡Completada!

En esta fase, nos enfocamos en mejorar la base del proyecto, la estructura del código y la experiencia de usuario en cuanto a la gestión de archivos.

- [x] **Refactorización de la Estructura de Código:**
    - [x] **Dividir `main.py`:** Extraer la lógica de cada paso del pipeline en funciones o clases dedicadas dentro de sus respectivos módulos en `Components/`. `main.py` debería ser principalmente un orquestador de alto nivel.
    - [x] **Consolidar Utilidades:** Revisar `Components/SafeName.py` y las funciones de slug/date prefix en `Components/YoutubeDownloader.py` para crear una única utilidad de nombrado seguro en `Components/common_utils.py` (o similar).
    - [x] **Eliminar Código Obsoleto/No Usado:** Remover `Components/Speaker.py` y `Components/SpeakerDetection.py` ya que sus funcionalidades han sido reemplazadas por `FaceCropYOLO.py` y la diarización de `pyannote`.
    - [x] **Revisar `Components/Edit.py` y `Components/common_ffmpeg.py`:** Asegurar que las funciones de FFmpeg estén organizadas de la manera más lógica, quizás moviendo `_ffmpeg_path` y `_ensure_parent` a `common_ffmpeg.py` si no están ya allí, y asegurando que `Edit.py` solo contenga funciones de edición de alto nivel.

- [ ] **Gestión de Archivos de Salida:**
    - [ ] **Manejo de Sobrescritura:** Implementar una lógica en `main.py` (o en las funciones de guardado de cada componente) que pregunte al usuario si desea sobrescribir los archivos existentes si se intenta procesar el mismo video nuevamente. Alternativamente, generar un sufijo único (e.g., `_run2`) si el usuario no desea sobrescribir.
    - [ ] **Estructura de Carpetas de Salida para Debugging:** Asegurar que la estructura de `out/<video_name>/` y `work/<video_name>/` permita almacenar todos los archivos intermedios generados en el proceso, facilitando el debugging y el seguimiento del flujo de cada video. Considerar opciones como:
        - [ ] Un solo directorio de trabajo/salida por video para simplificar la gestión y consolidar todos los archivos generados.
    - [ ] **Nombres de Archivo Consistentes:** Asegurar que todos los archivos generados dentro de la carpeta de un video sigan una convención de nombrado consistente y fácil de entender.

#### Fase 2: Inteligencia y Personalización Mejoradas (v0.3)

Esta fase se centrará en añadir más "inteligencia" al proceso de edición y permitir una mayor personalización del resultado final.

- [x] **Mejoras de IA:** Auto-generación de títulos, descripciones y hashtags para los clips.
- [ ] **Análisis de Contenido Avanzado:** Detección de temática, tono y otros metadatos relevantes del video para una edición más inteligente.
    - [ ] **Detección de Picos Emocionales:** Identificar momentos de alta emoción (risa, sorpresa, enojo, etc.) en audio/video.
    - [ ] **Análisis de Sentimiento del Texto:** Clasificar el tono del segmento (humorístico, serio, etc.).
- [ ] **Optimización de Ganchos (Hooks):**
    - [ ] **Generación de Ganchos A/B Testing:** Crear múltiples variaciones de intros/ganchos.
    - [ ] **Identificación de Preguntas/Afirmaciones Clave:** Extraer frases impactantes para ganchos.
- [ ] **Efectos Visuales Dinámicos (Basados en Contenido):**
    - [ ] **Resaltado de Palabras Clave Visual:** Cambiar color/tamaño/efecto de palabras importantes en subtítulos.
    - [ ] **Animaciones de Texto (Kinetic Typography):** Aplicar animaciones sutiles a subtítulos.
    - [ ] **Detección de Gestos/Expresiones Faciales:** (Más avanzado) Usar para activar efectos visuales.
- [ ] **Emojis Inteligentes:** Inserción automática de emojis relevantes (💡, 😂, 💰) en los subtítulos para aumentar el engagement.
- [ ] **Modo Karaoke:** Opción para resaltar palabra por palabra en los subtítulos a medida que se pronuncian.
- [ ] **Personalización:** Permitir configurar fácilmente el estilo de los subtítulos (fuentes, colores, etc.) a través de un archivo de configuración.

#### Edición Automática Avanzada y Profesional (Transversal)

Desarrollo de algoritmos sofisticados para lograr una edición de video automática de calidad profesional, adaptándose a diversos tipos de contenido (entrevistas multi-speaker, conferencias) y optimizando el engagement visual.

- [ ] **Algoritmos de Edición Contextual:**
    - [ ] **Detección de Escenas y Eventos Clave:** Identificar cambios de tema, preguntas/respuestas, énfasis, etc.
    - [ ] **Edición Basada en Diálogo:** Priorizar visibilidad del hablante activo, transiciones suaves y encuadres dinámicos.
    - [ ] **Manejo de Múltiples Hablantes:** Lógicas de cámara para alternar, planos grupales y gestión de entrada/salida.
    - [ ] **Optimización de Ritmo y Flujo:** Ajustar duración de planos y velocidad de transiciones para engagement.
- [ ] **Transiciones Automáticas Inteligentes:**
    - [ ] **Selección de Transiciones Dinámicas:** Elegir tipo de transición adecuado al contexto (corte, disolvencia).
    - [ ] **Transiciones de Cámara Suaves y Profesionales:** Mejorar paneo y zoom para calidad cinematográfica.
- [ ] **Integración de Elementos Visuales y Sonoros:**
    - [ ] **B-Roll y Material de Apoyo:** (Muy avanzado) Sugerir/insertar automáticamente material de archivo relevante.
    - [ ] **Diseño Sonoro Adaptativo:** Ajustar volumen de música, añadir efectos de sonido sutiles, mejorar claridad del habla.
- [ ] **Aprendizaje y Adaptación:**
    - [ ] **Feedback Loop para Edición:** (A largo plazo) Aprender de preferencias del usuario o rendimiento del video.

#### Fase 3: Automatización y UI (v0.4+)

La fase final se enfocará en la automatización completa del flujo de trabajo y la creación de una interfaz de usuario amigable.

- [ ] **Generación Dinámica de Intros:** Creación de intros atractivas con frases de "gancho" (ej. "¡No vas a creer lo que dijo X Persona!"), usando síntesis de voz (TTS) y música de fondo sin copyright.
    - [ ] **Integración de Música Dinámica:** Selección de música por tono y ajuste de volumen dinámico.
    - [ ] **Generación de Miniaturas (Thumbnails) Atractivas:** Identificar frames clave y superponer texto/elementos.
- [ ] **Publicación Automática:** Integración con APIs de YouTube y Facebook para la publicación directa de los videos generados.
- [ ] **Soporte para Múltiples Idiomas:** Expandir TTS y subtítulos a varios idiomas.
- [ ] **UI Web:** Crear una interfaz gráfica con Gradio o Streamlit para un uso no técnico.

---

## 🤝 Contribuciones

¡Se aceptan PRs! 🙌  
Si quieres colaborar, abre un *issue* o crea un *pull request*.

---

## 📜 Licencia

[MIT](./LICENSE) © 2025