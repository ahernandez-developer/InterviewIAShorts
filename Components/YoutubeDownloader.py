import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pytubefix import YouTube
import ffmpeg
import re

def clean_filename(s: str) -> str:
    # elimina caracteres ilegales en nombres de Windows
    return re.sub(r'[<>:"/\\|?*]', '', s)

def get_video_size(stream):
    return stream.filesize / (1024 * 1024)

def download_youtube_video(url):
    try:
        yt = YouTube(url)
        raw_title = yt.title
        safe_title = clean_filename(raw_title)

        video_streams = yt.streams.filter(type="video").order_by('resolution').desc()
        audio_stream = yt.streams.filter(only_audio=True).first()

        print("Available video streams:")
        for i, stream in enumerate(video_streams):
            size = get_video_size(stream)
            stream_type = "Progressive" if stream.is_progressive else "Adaptive"
            print(f"{i}. Resolution: {stream.resolution}, Size: {size:.2f} MB, Type: {stream_type}")

        choice = int(input("Enter the number of the video stream to download: "))
        selected_stream = video_streams[choice]

        os.makedirs('videos', exist_ok=True)

        print(f"Downloading video: {raw_title!r}")
        # Descarga con nombre temporal
        video_file = selected_stream.download(
            output_path='videos',
            filename_prefix="video_"
        )
        # Renombrar para quitar caracteres ilegales
        base, ext = os.path.splitext(video_file)
        new_video_file = os.path.join('videos', f"video_{safe_title}{ext}")
        os.replace(video_file, new_video_file)
        video_file = new_video_file

        if not selected_stream.is_progressive:
            print("Downloading audio...")
            audio_file = audio_stream.download(
                output_path='videos',
                filename_prefix="audio_"
            )
            base, ext = os.path.splitext(audio_file)
            new_audio_file = os.path.join('videos', f"audio_{safe_title}{ext}")
            os.replace(audio_file, new_audio_file)
            audio_file = new_audio_file

            print("Merging video and audio...")
            output_file = os.path.join('videos', f"{safe_title}.mp4")
            stream = ffmpeg.input(video_file)
            audio = ffmpeg.input(audio_file)
            merged = ffmpeg.output(
                stream, audio,
                output_file,
                vcodec='libx264',
                acodec='aac',
                strict='experimental'
            )
            ffmpeg.run(merged, overwrite_output=True)

            # limpias los temporales
            os.remove(video_file)
            os.remove(audio_file)
        else:
            # en progresivos ya está en mp4
            output_file = video_file

        print(f"Downloaded: {raw_title!r} to 'videos' as {os.path.basename(output_file)}")
        return output_file

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure you have the latest version of pytube and ffmpeg-python installed:")
        print("    pip install --upgrade pytube ffmpeg-python")
        print("Also ensure that ffmpeg is on your PATH.")

if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    download_youtube_video(youtube_url)
