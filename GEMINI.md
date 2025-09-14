# Notas para el Agente Gemini Futuro

Este documento contiene un resumen técnico y el rationale detrás de las decisiones de diseño para que puedas continuar el desarrollo de este proyecto de forma eficiente.

**Alias del Proyecto:** Generador de Shorts Virales con IA

---

## v0.2 - Inteligencia de Contenido y Multi-Highlight

La versión 0.2 representa un salto arquitectónico importante, haciendo el pipeline más inteligente, robusto y potente. Pasamos de generar un solo clip a ser un sistema que analiza el tipo de contenido y genera múltiples clips optimizados.

### 1. Migración a Google Gemini y Schemas Pydantic

- **Problema:** La biblioteca de `openai` tuvo cambios que rompieron la compatibilidad. Además, se buscaba una integración más profunda con el ecosistema de Google.
- **Solución:** Se migró toda la lógica de LLM de `openai` a `google-generativeai` (Gemini).
- **Mejora Clave:** Para asegurar que la IA siempre devuelva un JSON con el formato esperado, se introdujo el uso de **`pydantic`**. Ahora, cada llamada a la API de Gemini se configura con un `response_schema` que define la estructura de salida. Esto elimina la necesidad de analizar strings y reduce drásticamente los errores de formato, haciendo el sistema mucho más fiable.

### 2. Clasificador de Contenido Multimodal

- **Problema:** El pipeline trataba todos los videos por igual (como entrevistas), lo cual no es óptimo para, por ejemplo, escenas de series o documentales.
- **Solución:** Se creó un nuevo componente, `Components/ContentClassifier.py`.
- **Arquitectura:**
    1.  **Análisis Multimodal:** Este componente no solo lee la transcripción, sino que también **extrae 5 fotogramas representativos** del video.
    2.  **Clasificación con Gemini:** Envía el título, la transcripción y las imágenes a `gemini-1.5-flash` para clasificar el video en una de tres categorías: `interview`, `presentation` o `general_content`.
    3.  **Lógica Condicional:** El pipeline principal ahora usa esta clasificación para decidir si debe aplicar la lógica de seguimiento de rostros (`interview`, `presentation`) o usar un enfoque de edición más general.

### 3. Pipeline Generativo de Múltiples Highlights

- **Problema:** El sistema anterior solo generaba un clip. Además, si la IA no encontraba un segmento que cumpliera las estrictas reglas de duración (50-70s), fallaba y no producía nada.
- **Solución:** Se rediseñó por completo el proceso de selección de highlights.
- **Nueva Arquitectura de Dos Pasos:**
    1.  **La IA Sugiere (sin restricciones):** Se modificó el prompt en `LanguageTasks.py`. Ahora se le pide a Gemini que haga lo que mejor sabe hacer: encontrar los **3 momentos más interesantes** del video, sin preocuparse por la duración.
    2.  **El Código Ajusta (con precisión):** Se implementó una nueva función, `_adjust_highlight_duration`, que recibe las sugerencias de la IA. Esta función, de forma programática y usando los timestamps a nivel de palabra, **expande o contrae** cada segmento sugerido hasta que encaje perfectamente en el rango de 50-70 segundos.
- **Procesamiento en Bucle:** El pipeline principal ahora itera sobre la lista de highlights ajustados y **ejecuta todo el proceso de edición para cada uno**, generando múltiples videos finales.
- **Estructura de Salida:** Los videos resultantes se guardan en subcarpetas numeradas (`highlight_1`, `highlight_2`, etc.) para mantener el orden.

---

## v0.1 - Resumen del Proyecto (Estado Anterior)

El objetivo inicial era transformar un video largo de YouTube en un clip corto vertical (9:16) con subtítulos dinámicos. La v0.1 fue la primera versión estable de este flujo.

(... El resto del contenido de v0.1 se mantiene como referencia histórica ...)

---

## Roadmap (Actualizado)

El proyecto ha evolucionado. La fase de inteligencia de contenido se ha completado, y ahora el foco principal vuelve a ser la calidad cinematográfica de la cámara virtual.

#### Fase 1.2: Inteligencia de Contenido y Multi-Highlight (v0.2) - ¡Completada!

- [x] **Migración a Google Gemini:** Reemplazo de OpenAI por Gemini para todas las tareas de LLM.
- [x] **Respuestas JSON Robustas:** Implementación de `pydantic` para forzar esquemas de salida en las respuestas de la IA.
- [x] **Clasificador de Contenido Multimodal:** Creación de `ContentClassifier.py` que analiza video y texto para determinar el tipo de contenido.
- [x] **Pipeline Condicional:** El flujo de trabajo se adapta según el video sea una `entrevista`, `presentación` o `contenido general`.
- [x] **Generación de Múltiples Highlights:** El sistema ahora identifica los 3 mejores momentos y los procesa en videos separados.
- [x] **Ajuste Programático de Duración:** Se implementó una lógica determinista para expandir/contraer los clips a la duración deseada, eliminando los fallos del LLM por restricciones de duración.

#### Fase 1.5: Calidad de Edición Profesional (Prioridad Actual)

*El objetivo de esta fase es asegurar que cada clip generado cumpla con un estándar de calidad profesional, con una edición visual de alto impacto.*

- [ ] **Cámara Virtual Profesional:** Re-diseño del sistema de cámara para lograr un acabado de alta calidad, inspirado en operadores de cámara humanos.
    - [ ] **Fase 1 (Prioridad Inmediata): Movimiento Orgánico con Simulación Física:**
        - Implementar un modelo de **muelle-amortiguador (spring-damper)** para las transiciones de cámara (paneo y zoom).
        - **Objetivo:** Lograr un movimiento natural con aceleración y desaceleración suaves, eliminando la sensación robótica y permitiendo un control fino sobre la "sensación" del movimiento.
    - [ ] **Fase 2: Composición Inteligente:**
        - Implementar la **Regla de los Tercios** para posicionar a los hablantes, creando encuadres más dinámicos y visualmente atractivos.
        - Añadir un sutil efecto de **"respiración" (breathing)** con zoom lento en monólogos largos para evitar tomas estáticas.
    - [ ] **Fase 3: Ritmo Consciente del Contexto:**
        - Desarrollar un **paneo anticipatorio** que mueva la cámara lentamente hacia el próximo hablante *antes* de que intervenga.
        - Implementar una **duración de transición dinámica** que se adapte al ritmo de la conversación (transiciones rápidas para diálogos ágiles, lentas para pausas reflexivas).

### v0.2.1 - Estabilidad y Mejoras de UX (Completada)

Esta versión se centró en resolver errores críticos y mejorar la experiencia de usuario a través de barras de progreso más fiables y consistentes.

- [x] **Robustez en la Transcripción:** Se corrigió el paso de argumentos a `transcribeAudio` para asegurar el uso correcto del tamaño del modelo y la ruta de guardado del JSON de voz, eliminando `NameError`s.
- [x] **Flujo de Generación de Metadatos Refinado:** Se ajustó la llamada a `generate_video_metadata`, moviéndola al bucle de procesamiento de highlights y asegurando que reciba el texto de highlight correcto, resolviendo `TypeError`s y alineándose con la estrategia de metadatos por highlight.
- [x] **Barras de Progreso Precisas y Consistentes:**
    - Se habilitó el reporte de progreso exacto para las operaciones de muxing de `ffmpeg` calculando y pasando correctamente la `total_duration` a `run_ffmpeg_with_progress`.
    - Se mejoró el parseo del progreso de `ffmpeg` en `run_ffmpeg_with_progress` para incluir el formato de salida `time=`, lo que resulta en actualizaciones más frecuentes y fiables.
    - Se estandarizó el estilo de las barras de progreso reemplazando `tqdm` por `rich.progress` en el paso de renderizado de video, ofreciendo una experiencia de usuario uniforme.
- [x] **Gestión de Dependencias:** Se aseguró la importación correcta de `run_ffmpeg_with_progress` en `Components/FaceCropYOLO.py` para resolver `NameError`s.

(... El resto del roadmap futuro se mantiene ...)

---

## Cómo Ejecutar

El script es interactivo.

```bash
python main.py
```