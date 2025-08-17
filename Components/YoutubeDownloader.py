# Components/YoutubeDownloader.py  (v3 con subcarpetas por video)
import os, re, shutil, subprocess
from pathlib import Path

from Components.SafeName import safe_name
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pytubefix import YouTube

def clean_filename(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '', s)

def get_video_size(stream):
    try:
        return stream.filesize / (1024 * 1024)
    except Exception:
        return 0.0

def _ffmpeg_path() -> str:
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ff = shutil.which(exe)
    if not ff:
        raise EnvironmentError("FFmpeg no encontrado en PATH.")
    return ff

def _has_nvenc() -> bool:
    ff = _ffmpeg_path()
    proc = subprocess.run([ff, "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return "h264_nvenc" in proc.stdout

def _venc_flags():
    return ["-c:v", "h264_nvenc", "-preset", "p5"] if _has_nvenc() else ["-c:v", "libx264", "-preset", "veryfast"]

def _run(cmd):
    print("[ffmpeg] " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"FFmpeg error ({p.returncode}):\n{p.stdout}")
    tail = "\n".join(p.stdout.splitlines()[-10:])
    if tail.strip():
        print(tail)

def download_youtube_video(url):
    try:
        yt = YouTube(url)
        raw_title = yt.title
        base_name  = safe_name(raw_title)

        video_streams = yt.streams.filter(type="video").order_by('resolution').desc()
        audio_stream = yt.streams.filter(only_audio=True).first()

        print("Available video streams:")
        for i, stream in enumerate(video_streams):
            size = get_video_size(stream)
            stream_type = "Progressive" if stream.is_progressive else "Adaptive"
            print(f"{i}. Resolution: {stream.resolution}, Size: {size:.2f} MB, Type: {stream_type}")

        choice = int(input("Enter the number of the video stream to download: "))
        selected_stream = video_streams[choice]

        # === nueva subcarpeta por video ===
        base_dir = Path('videos')
        video_dir = base_dir / base_name
        video_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading video: {raw_title!r}")
        tmp_video = selected_stream.download(output_path=str(video_dir), filename_prefix="video_")
        base, ext = os.path.splitext(tmp_video)
        new_video_file = video_dir / f"video_{base_name}{ext}"
        os.replace(tmp_video, new_video_file)
        video_file = str(new_video_file)

        if not selected_stream.is_progressive:
            print("Downloading audio...")
            tmp_audio = audio_stream.download(output_path=str(video_dir), filename_prefix="audio_")
            base, aext = os.path.splitext(tmp_audio)
            new_audio_file = video_dir / f"audio_{base_name}{aext}"
            os.replace(tmp_audio, new_audio_file)
            audio_file = str(new_audio_file)

            print("Merging video and audio...")
            ff = _ffmpeg_path()
            venc = _venc_flags()
            output_file = video_dir / f"{base_name}.mp4"

            cmd = [
                ff, "-y",
                "-hwaccel", "auto",
                "-i", video_file,
                "-i", audio_file,
                "-map", "0:v:0", "-map", "1:a:0",
                *venc,
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart",
                str(output_file),
            ]
            _run(cmd)

            # limpia temporales
            os.remove(video_file)
            os.remove(audio_file)
            final_path = str(output_file)
        else:
            final_path = video_file

        print(f"Downloaded: {raw_title!r} to {video_dir} as {os.path.basename(final_path)}")
        return final_path

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Try: pip install --upgrade pytubefix ffmpeg-python")
        return None

if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    download_youtube_video(youtube_url)
