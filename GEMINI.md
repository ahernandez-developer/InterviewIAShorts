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

5.  **Corrección de la Relación de Aspecto del Recorte (Eliminación de Estiramiento)**
    - **Problema:** El video final mostraba estiramientos o compresiones visuales, especialmente durante los movimientos de cámara.
    - **Causa:** La ventana de recorte (`win`) generada por `FaceCropYOLO.py` no siempre mantenía una relación de aspecto 9:16 (vertical) antes de ser redimensionada a la resolución final (1080x1920). Esto causaba distorsión al forzar una imagen con una relación de aspecto incorrecta en un marco 9:16.
    - **Solución:** Se refactorizó la lógica de recorte en `FaceCropYOLO.py` para asegurar que `crop_width` y `crop_height` siempre se calculen de forma que mantengan estrictamente la relación de aspecto 9:16. Se ajustaron las coordenadas de recorte (`left`, `top`, `right`, `bottom`) y la lógica de recorte para garantizar que la región `win` siempre tenga la proporción correcta antes del redimensionamiento final.

6.  **Seguimiento Robusto del Hablante y Centrado de Cámara**
    - **Problema:** La cámara a veces apuntaba a "nada" (el centro del encuadre) en lugar de al hablante, especialmente durante los cambios de hablante.
    - **Causa:** Existía una falta de coincidencia entre la línea de tiempo del `speech.json` (generado para el video completo original) y el `tsec` (tiempo actual en el video recortado). Además, la función `_find_turn_index` era demasiado estricta y devolvía `-1` (no se encontró turno) si `tsec` caía en pequeños huecos entre los turnos de hablante (creados por la compactación de `_load_turns`).
    - **Solución:**
        1.  Se modificó `FaceCropYOLO.py` para aceptar `highlight_start_sec` (el inicio del segmento recortado del video original).
        2.  La función `_load_turns` se actualizó para restar `highlight_start_sec` de todos los timestamps de los turnos, alineando así el `speech.json` con la línea de tiempo del video recortado.
        3.  La función `_find_turn_index` se hizo más robusta introduciendo una `tolerance` (0.15 segundos) en su búsqueda binaria. Ahora, si `tsec` no cae exactamente dentro de un turno, pero está muy cerca de uno (dentro de la tolerancia), se considera ese turno activo, evitando que la cámara se centre en la nada debido a pequeños huecos.

---

## Estado Actual y Próximos Pasos

El proyecto se encuentra en la **versión 0.1**. Es funcional y estable.

### Roadmap (Priorizado)

#### Fase 1.5: Calidad de Edición Profesional (Prioridad Actual)

*El objetivo de esta fase es asegurar que cada clip generado cumpla con un estándar de calidad profesional, con una duración consistente y una edición visual de alto impacto.*

- [x] **Consistencia en la Duración del Clip:**
    - [x] Modificar el prompt y la lógica en `LanguageTasks.py` para instruir al LLM a seleccionar un *highlight* que dure estrictamente **entre 50 y 70 segundos**.
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

## Lecciones Aprendidas y Metodología de Debugging

Durante el desarrollo de la v0.1 y las mejoras de la Fase 1.5, se consolidó una metodología de debugging iterativa que resultó crucial para resolver problemas complejos de procesamiento de video. Para futuros agentes, es vital comprender y aplicar este enfoque:

1.  **Diagnóstico Basado en Observación:** Siempre iniciar con una descripción clara y detallada del problema visual o funcional reportado por el usuario.
2.  **Hipótesis y Localización del Problema:** Formular hipótesis sobre la causa raíz y usar el conocimiento de la arquitectura del pipeline (`docs/Workflow.md`) para identificar el componente o la sección de código más probable.
3.  **Instrumentación con Debugging:** Insertar `logging.info` o `logging.debug` estratégicamente para exponer variables clave, flujos de ejecución y estados intermedios. Esto es fundamental para transformar una observación visual en datos concretos.
    *   **Ejemplo:** Para el problema de "estiramiento", se imprimió la relación de aspecto de la ventana de recorte (`win.shape[1] / win.shape[0]`) para confirmar que no era 9:16.
    *   **Ejemplo:** Para el problema de "cámara a la nada", se imprimieron `tsec`, `last_face`, `target_anchor` y los detalles de la búsqueda de turnos (`DEBUG_TURN_FINDER`) para identificar el desajuste de timestamps y los huecos.
4.  **Análisis de Datos de Debugging:** Interpretar la salida de debugging para validar o refutar la hipótesis inicial y refinar la comprensión del problema.
5.  **Propuesta de Solución y Justificación:** Basado en el análisis, proponer una solución concreta y explicar *por qué* se espera que resuelva el problema, haciendo referencia a los datos de debugging.
6.  **Implementación Iterativa:** Aplicar los cambios de código de forma incremental, verificando cada paso.
7.  **Verificación y Feedback:** Ejecutar el pipeline y obtener feedback del usuario sobre la resolución del problema. Si persiste, volver al paso 1 con una hipótesis refinada.
8.  **Limpieza:** Una vez resuelto el problema, eliminar todos los prints de debugging y las importaciones temporales para mantener el código limpio y eficiente.

**Importancia de la Sincronización de Timestamps:** Un aprendizaje clave fue la criticidad de la sincronización de timestamps entre diferentes etapas del pipeline (video original, video recortado, `speech.json`). Cualquier desalineación, por pequeña que sea, puede tener efectos cascada en la precisión del seguimiento y la generación de subtítulos.

**Robustez ante Datos Imperfectos:** La necesidad de hacer que la lógica de búsqueda de turnos (`_find_turn_index`) fuera tolerante a pequeños huecos en los datos de `speech.json` (debido a la compactación de turnos) subraya la importancia de construir algoritmos robustos que puedan manejar las imperfecciones inherentes a los datos del mundo real.

---

## Cómo Ejecutar

El script es interactivo.

```bash
python main.py
```
