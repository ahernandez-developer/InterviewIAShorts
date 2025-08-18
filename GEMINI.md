# Notas para el Agente Gemini Futuro

Este documento contiene un resumen técnico y el rationale detrás de las decisiones de diseño para que puedas continuar el desarrollo de este proyecto de forma eficiente.

**Alias del Proyecto:** Generador de Shorts Virales con IA

---

## v0.1 - Resumen del Proyecto

El objetivo es transformar un video largo de una URL de YouTube en un clip corto vertical (9:16) con subtítulos dinámicos de alta calidad, listo para redes sociales. La versión 0.1 es la primera versión estable que cumple con este flujo de principio a fin.

El usuario actual (`oscar`) ha guiado el desarrollo de forma iterativa, enfocándose en la calidad del resultado final, especialmente en la sincronización y el aspecto visual de los subtítulos y la cámara.

---

## Arquitectura y Flujo de Trabajo

El proyecto sigue un pipeline modular orquestado por `main.py`. Los componentes principales residen en la carpeta `Components/`.

Para un análisis exhaustivo del flujo, **consulta `docs/Workflow.md`**. Contiene un diagrama y una descripción detallada de cada uno de los 9 pasos del pipeline.

**Resumen del Flujo:**
1.  **Descarga:** `YoutubeDownloader.py` baja el video.
2.  **Audio:** `Edit.py` extrae el audio a `.wav`.
3.  **Transcripción:** `Transcription.py` usa `faster-whisper` para obtener texto y **timestamps a nivel de palabra**, y opcionalmente `pyannote` para diarización. Guarda el resultado en `work/.../speech.json`.
4.  **Highlight:** `LanguageTasks.py` usa un LLM para decidir qué parte del video cortar.
5.  **Corte:** `Edit.py` corta el video de forma precisa (ver Rationale).
6.  **Cámara Virtual:** `FaceCropYOLO.py` recorta a 9:16 y aplica paneo/zoom suave.
7.  **Muxing:** Se une el video vertical con el audio del corte.
8.  **Subtítulos:** `Subtitles.py` genera un archivo `.ass` con subtítulos dinámicos.
9.  **Burn-in:** Se incrustan los subtítulos en el video final.

---

## Decisiones Técnicas Clave (Rationale)

Estas son las decisiones importantes que se tomaron y el porqué. Entenderlas es clave para no introducir regresiones.

1.  **Corte de Video con Re-codificación (`copy=False`)**
    - **Problema:** Los subtítulos estaban desincronizados.
    - **Causa:** Al usar el modo de copia de stream (`-c copy`) en `ffmpeg`, los cortes solo pueden ocurrir en *keyframes*. Esto causaba un desfase entre el inicio real del video y el timestamp esperado, rompiendo la sincronización del audio y los subtítulos.
    - **Solución:** Se fuerza siempre la re-codificación (`copy=False`) en la función `trim_video_ffmpeg`. Esto es un poco más lento pero garantiza un corte preciso a nivel de frame, lo cual es **crítico** para la integridad del pipeline.

2.  **Transición Suave en Cámara Estática**
    - **Problema:** Al inicio del video, había un "salto" brusco en el encuadre.
    - **Causa:** El sistema pasaba de un encuadre por defecto a un encuadre anclado al primer hablante de forma instantánea.
    - **Solución:** Se implementó una rampa de transición de 0.5 segundos en `FaceCropYOLO.py`. Ahora, al cambiar de objetivo (o al encontrar el primero), la cámara se mueve suavemente usando una curva de interpolación *ease-in-out*.

3.  **Agrupación de Subtítulos por Pausas**
    - **Problema:** Los subtítulos iniciales se cortaban por número de palabras, lo que resultaba en un ritmo antinatural.
    - **Solución:** La lógica en `generate_ass` (`Subtitles.py`) fue reescrita. Ahora, el principal criterio para cortar una línea de subtítulo es detectar una pausa en el habla de más de `0.35` segundos. Esto hace que los subtítulos fluyan de manera mucho más orgánica y profesional.

4.  **Uso de `faster-whisper` y Timestamps por Palabra**
    - Se eligió `faster-whisper` por su eficiencia (velocidad y uso de memoria) frente a la implementación original de OpenAI.
    - La activación de `word_timestamps=True` fue la decisión clave que habilitó la creación de subtítulos dinámicos de alta calidad. El `speech.json` resultante es el corazón de la sincronización del proyecto.

---

## Estado Actual y Próximos Pasos

El proyecto se encuentra en la **versión 0.1**. Es funcional y estable.

Los próximos pasos lógicos están documentados en el `README.md` y se centran en añadir más "inteligencia" y personalización:

- **Emojis Inteligentes:** Usar el LLM para añadir emojis relevantes al final de las líneas de subtítulos.
- **Modo Karaoke Real:** Implementar el resaltado palabra por palabra. Esto requeriría modificar la generación del archivo `.ass` para usar tags de karaoke como `{\k...}`.
- **UI Web:** Abstraer la complejidad en una interfaz gráfica simple.

---

## Cómo Ejecutar

El script es interactivo.

```bash
python main.py
```
