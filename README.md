# AI Shorts Generator 🎬🤖 v0.1

> Generador automático de **YouTube Shorts** a partir de videos largos usando Whisper, OpenAI GPT y OpenCV/FFmpeg.  
> Convierte entrevistas o podcasts en clips atractivos en formato vertical listos para subir a TikTok, Reels y Shorts.

---

## 🚀 Características (v0.1)

- 📥 **Descarga de Video:** Soporte para YouTube y otras fuentes mediante `yt-dlp`.
- 🎙️ **Transcripción Precisa:** Transcripción automática de alta calidad con `faster-whisper`, con timestamps a nivel de palabra.
- 🧠 **Selección Inteligente:** Uso de **GPT-4** para analizar la transcripción y encontrar el segmento más "viral" o interesante del video.
- 🗣️ **Diarización de Hablantes:** Detección de quién habla y cuándo, permitiendo crear encuadres fijos por hablante.
- 🎥 **Cámara Virtual Inteligente:**
    - **Modo Estático:** Crea planos fijos por hablante con transiciones suaves entre ellos, simulando un cambio de cámara profesional.
    - **Modo Dinámico:** Si no hay turnos de habla, una cámara virtual sigue al sujeto con paneo y zoom cinematográfico.
- 🔥 **Subtítulos Dinámicos:**
    - Generación automática de subtítulos en formato `.ass`.
    - **Sincronización por Pausas:** Los subtítulos se agrupan y aparecen en pantalla siguiendo el ritmo natural del habla y las pausas del hablante.
    - **Estilo Viral:** Fuente de alto impacto (Impact), tamaño grande y contorno para máxima legibilidad en móviles.
- 🐳 **Soporte Docker:** Compatible con Docker para despliegue en cualquier entorno.

---

## 📦 Requisitos

- **Python 3.8+**
- **FFmpeg** instalado y accesible desde la terminal (`ffmpeg -version`)
- **Cuenta de OpenAI** y API Key activa

---

## ⚙️ Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/ahernandez-developer/InterviewIAShorts.git
cd InterviewIAShorts

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate   # en Linux/Mac
venv\Scripts\activate    # en Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
# (Crea un archivo .env a partir de .env.example y añade tu API key de OpenAI)
```

---

## ▶️ Uso

El script ahora es interactivo. Simplemente ejecútalo desde la raíz del proyecto:

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
Adicionalmente, la carpeta `work/` contiene archivos intermedios como el audio, la transcripción completa (`speech.json`), etc.

---

## 📊 Roadmap

### Logros de la v0.1

- [x] Soporte para `faster-whisper` para transcripciones rápidas y precisas.
- [x] Generación de subtítulos dinámicos en formato `.ass`.
- [x] Agrupación de subtítulos por pausas naturales del habla.
- [x] Cámara estática con transiciones suaves entre hablantes.
- [x] Lógica de recorte de video precisa para evitar desincronización.

### Próximos Pasos (v0.2 y más allá)

- [ ] **Emojis Inteligentes:** Inserción automática de emojis relevantes (💡, 😂, 💰) en los subtítulos para aumentar el engagement.
- [ ] **Modo Karaoke:** Opción para resaltar palabra por palabra en los subtítulos a medida que se pronuncian.
- [ ] **UI Web:** Crear una interfaz gráfica con Gradio o Streamlit para un uso no técnico.
- [ ] **Mejoras de IA:** Auto-generación de títulos, descripciones y hashtags para los clips.
- [ ] **Personalización:** Permitir configurar fácilmente el estilo de los subtítulos (fuentes, colores, etc.) a través de un archivo de configuración.

---

## 🤝 Contribuciones

¡Se aceptan PRs! 🙌  
Si quieres colaborar, abre un *issue* o crea un *pull request*.

---

## 📜 Licencia

[MIT](./LICENSE) © 2025