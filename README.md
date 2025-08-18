# AI Shorts Generator 🎬🤖

> Generador automático de **YouTube Shorts** a partir de videos largos usando Whisper, OpenAI GPT y OpenCV/FFmpeg.  
Convierte entrevistas o podcasts en clips atractivos en formato vertical listos para subir a TikTok, Reels y Shorts.

---

## 🚀 Características

- 📥 Descarga de videos desde YouTube u otras fuentes (`yt-dlp`).
- 📝 Transcripción automática con **Whisper** (soporte multi-idioma).
- ✂️ Selección inteligente de highlights con **GPT-4**.
- 🧑‍🤝‍🧑 Detección de speakers y segmentación por turnos de voz.
- 🎥 Recorte automático a formato **9:16 vertical**, centrando en rostros o áreas relevantes.
- 📂 Genera un **manifest.json** por corrida con metadatos (timestamps, prompts, scores, modelos usados).
- 🐳 Compatible con Docker para despliegue en cualquier entorno.

---

## 📦 Requisitos

- **Python 3.8+**
- **FFmpeg** instalado y accesible desde la terminal (`ffmpeg -version`)
- **OpenCV** con soporte de modelos Haar/SSD
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
cp .env.example .env
# Editar .env y colocar tu API key de OpenAI
```

---

## 🔑 Configuración

Archivo `.env`:

```ini
OPENAI_API_KEY=tu_api_key_aqui
```

Ejemplo de `.env.example` ya incluido en el repo.

---

## ▶️ Uso

Ejecutar el pipeline principal desde la raíz del proyecto:

```bash
python main.py --url "https://youtube.com/watch?v=XXXX" --max-clips 5 --outdir ./outputs
```

### Parámetros disponibles

| Parámetro       | Descripción |
|-----------------|-------------|
| `--url`         | URL del video de YouTube |
| `--max-clips`   | Número máximo de clips a generar |
| `--outdir`      | Carpeta de salida (default: `./outputs`) |
| `--model-size`  | Tamaño del modelo Whisper (`tiny`, `base`, `small`, `medium`, `large`) |
| `--language`    | Forzar idioma de transcripción (`es`, `en`, etc.) |

---

## 📂 Estructura de salida

Cada corrida genera una carpeta con:

```
outputs/
└── <video_id>/
    ├── clips/
    │   ├── clip_01.mp4
    │   ├── clip_02.mp4
    │   └── ...
    ├── manifest.json
    ├── transcription.txt
    └── highlights.json
```

### Ejemplo `manifest.json`

```json
{
  "video_id": "abcd1234",
  "source_url": "https://youtube.com/watch?v=abcd1234",
  "created_at": "2025-08-17T12:00:00Z",
  "model": "gpt-4",
  "whisper_model": "medium",
  "clips": [
    {
      "file": "clip_01.mp4",
      "start": "00:01:23",
      "end": "00:02:45",
      "score": 0.91,
      "speaker": "Speaker 1",
      "highlight_text": "Explicación clave del invitado..."
    }
  ]
}
```

---

## 🐳 Uso con Docker

```bash
# Construir imagen
docker build -t ai-shorts .

# Ejecutar pipeline
docker run --rm -it \
    -v $(pwd)/outputs:/app/outputs \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    ai-shorts --url "https://youtube.com/watch?v=XXXX"
```

---

## 🔧 Troubleshooting

- **`ffmpeg: command not found`** → Instala FFmpeg:  
  - Ubuntu: `sudo apt install ffmpeg`  
  - Mac: `brew install ffmpeg`  
  - Windows: [descargar binarios](https://ffmpeg.org/download.html)

- **CUDA no disponible** → Whisper correrá en CPU (más lento).

- **API Key inválida** → Verifica tu `.env` y que la cuenta de OpenAI tenga créditos.

---

## 📊 Roadmap

- [ ] Soporte a `faster-whisper` para transcripciones más rápidas.
- [ ] Generación de subtítulos (`.srt`) y *burn-in* opcional.
- [ ] UI con Gradio para uso no técnico.
- [ ] Auto-generación de thumbnails y hashtags.

---

## 🤝 Contribuciones

¡Se aceptan PRs! 🙌  
Si quieres colaborar, abre un *issue* o crea un *pull request*.

---

## 📜 Licencia

[MIT](./LICENSE) © 2025
