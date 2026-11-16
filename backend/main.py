python
import os
import uuid
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# ==============================================================================
# KONFIGURASI & INISIALISASI (100% GEMINI AI - TANPA OPENAI)
# ==============================================================================
app = FastAPI(title="AutoClip AI Backend (100% Gemini Edition)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None
    print("PERINGATAN: GEMINI_API_KEY belum diatur di Environment Variables!")

BASE_DIR = Path("autoclip_storage")
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
MAX_FILE_SIZE_MB = 500


# ==============================================================================
# MODEL DATA
# ==============================================================================
class ClipMetadata(BaseModel):
    clip_index: int
    title: str
    reason: str
    start_time: float
    end_time: float
    subtitle_text: str
    video_url: str
    srt_url: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    clips: List[ClipMetadata]


class ReviseRequest(BaseModel):
    start_time: float
    end_time: float
    subtitle_text: str


# ==============================================================================
# FUNGSI UTILITAS (FFMPEG & SRT)
# ==============================================================================
def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_file(srt_path: Path, duration: float, subtitle_text: str):
    words = subtitle_text.strip().split()
    if not words:
        srt_content = f"1\n00:00:00,000 --> {format_srt_timestamp(duration)}\n \n"
        srt_path.write_text(srt_content, encoding="utf-8")
        return

    chunk_size = 8
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
    time_per_chunk = duration / len(chunks)

    srt_lines = []
    for idx, chunk in enumerate(chunks, start=1):
        t_start = (idx - 1) * time_per_chunk
        t_end = idx * time_per_chunk
        srt_lines.append(
            f"{idx}\n{format_srt_timestamp(t_start)} --> {format_srt_timestamp(t_end)}\n{chunk}\n"
        )

    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")


def render_clip_with_ffmpeg(
    source_video: Path,
    output_video: Path,
    srt_path: Path,
    start_time: float,
    end_time: float,
):
    duration = max(1.0, end_time - start_time)
    srt_escaped = str(srt_path.resolve()).replace("\\", "/").replace(":", "\\:")

    subtitle_style = (
        "FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=35"
    )

    cmd = [
        "ffmpeg", "-y", "-ss", f"{start_time:.3f}", "-i", str(source_video),
        "-t", f"{duration:.3f}", "-vf", f"subtitles='{srt_escaped}':force_style='{subtitle_style}'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output_video)
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg gagal memproses video: {result.stderr[-500:]}")


def extract_audio_mp3(source_video: Path, output_audio: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(source_video), "-vn",
        "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
        str(output_audio)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Gagal mengekstrak audio: {result.stderr[-500:]}")


# ==============================================================================
# ENDPOINT API
# ==============================================================================
@app.get("/")
async def health_check():
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    return {
        "status": "online",
        "service": "AutoClip AI Backend (100% Gemini Gratis)",
        "ffmpeg_installed": ffmpeg_ok,
        "gemini_configured": bool(GEMINI_API_KEY),
    }


@app.post("/process", response_model=JobResponse)
async def process_video(
    file: UploadFile = File(...),
    instruction: str = Form("Cari bagian paling menarik dan viral"),
    num_clips: int = Form(3),
    max_duration: int = Form(60),
):
    if not GEMINI_API_KEY or not gemini_model:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY belum diatur di server.")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Format video '{ext}' tidak didukung.")

    num_clips = max(1, min(10, num_clips))
    max_duration = max(8, min(180, max_duration))

    job_id = str(uuid.uuid4())
    job_folder = JOBS_DIR / job_id
    job_folder.mkdir(parents=True, exist_ok=True)

    source_video_path = job_folder / f"source{ext}"
    audio_path = job_folder / "extracted_audio.mp3"
    uploaded_gemini_file = None

    try:
        size_bytes = 0
        with open(source_video_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Ukuran video melebihi batas 500MB.")
                buffer.write(chunk)

        # 1. Ekstrak audio MP3 agar ringan dianalisis langsung oleh Gemini Multimodal
        extract_audio_mp3(source_video_path, audio_path)

        # 2. Unggah audio ke Gemini File API untuk transkripsi + pemilihan klip sekaligus
        uploaded_gemini_file = genai.upload_file(path=str(audio_path), mime_type="audio/mp3")
        while uploaded_gemini_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_gemini_file = genai.get_file(uploaded_gemini_file.name)

        prompt = (
            "Dengarkan file audio video ini dengan teliti. Tugas Anda sebagai editor video AI adalah:\n"
            "1. Menemukan momen percakapan paling menarik sesuai instruksi pengguna.\n"
            "2. Menentukan timestamp (start_time dan end_time dalam satuan detik) yang akurat tanpa memotong tengah kalimat.\n"
            "3. Menuliskan transkrip percakapan yang diucapkan pada klip tersebut untuk dijadikan subtitle.\n\n"
            f"Instruksi Pengguna: {instruction}\n"
            f"Jumlah klip maksimal: {num_clips}\n"
            f"Durasi maksimal tiap klip: {max_duration} detik (minimal 8 detik).\n\n"
            "Kembalikan HANYA format JSON valid dengan struktur:\n"
            '{"clips": [{"title": "Judul Singkat", "reason": "Alasan menarik", '
            '"start_time": 12.0, "end_time": 42.5, "subtitle_text": "Teks lengkap percakapan pada klip ini"}]}'
        )

        response = gemini_model.generate_content(
            [uploaded_gemini_file, prompt],
            generation_config={"response_mime_type": "application/json"}
        )

        ai_result = json.loads(response.text)
        selected_clips = ai_result.get("clips", [])

        if not selected_clips:
            raise HTTPException(status_code=422, detail="AI tidak menemukan klip yang sesuai. Coba instruksi lain.")

        # 3. Potong Video & Tanamkan Subtitle dengan FFmpeg
        output_clips: List[ClipMetadata] = []
        for idx, clip in enumerate(selected_clips[:num_clips]):
            start_t = float(clip["start_time"])
            end_t = float(clip["end_time"])
            if end_t - start_t > max_duration:
                end_t = start_t + max_duration
            if end_t <= start_t:
                end_t = start_t + 8.0

            sub_text = str(clip.get("subtitle_text", "")).strip()
            clip_video_path = job_folder / f"clip_{idx}.mp4"
            clip_srt_path = job_folder / f"clip_{idx}.srt"

            generate_srt_file(clip_srt_path, end_t - start_t, sub_text)
            render_clip_with_ffmpeg(source_video_path, clip_video_path, clip_srt_path, start_t, end_t)

            output_clips.append(
                ClipMetadata(
                    clip_index=idx,
                    title=clip.get("title", f"Klip #{idx + 1}"),
                    reason=clip.get("reason", "Momen pilihan Gemini AI"),
                    start_time=round(start_t, 2),
                    end_time=round(end_t, 2),
                    subtitle_text=sub_text,
                    video_url=f"/files/{job_id}/clip_{idx}.mp4",
                    srt_url=f"/files/{job_id}/clip_{idx}.srt",
                )
            )

        meta_path = job_folder / "metadata.json"
        meta_data = {
            "job_id": job_id,
            "source_video": source_video_path.name,
            "clips": [c.model_dump() for c in output_clips],
        }
        meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")

        return JobResponse(job_id=job_id, status="completed", clips=output_clips)

    except HTTPException:
        shutil.rmtree(job_folder, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(job_folder, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")
    finally:
        if audio_path.exists():
            audio_path.unlink()
        if uploaded_gemini_file:
            try:
                genai.delete_file(uploaded_gemini_file.name)
            except Exception:
                pass


@app.post("/revise/{job_id}/{clip_index}", response_model=ClipMetadata)
async def revise_clip(job_id: str, clip_index: int, payload: ReviseRequest):
    job_folder = JOBS_DIR / job_id
    meta_path = job_folder / "metadata.json"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Job tidak ditemukan.")

    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    source_video_path = job_folder / meta_data["source_video"]
    clips = meta_data["clips"]
    target_clip = next((c for c in clips if c["clip_index"] == clip_index), None)

    if not target_clip:
        raise HTTPException(status_code=404, detail="Klip tidak ditemukan.")

    try:
        clip_video_path = job_folder / f"clip_{clip_index}.mp4"
        clip_srt_path = job_folder / f"clip_{clip_index}.srt"

        duration = payload.end_time - payload.start_time
        generate_srt_file(clip_srt_path, duration, payload.subtitle_text)
        render_clip_with_ffmpeg(
            source_video_path, clip_video_path, clip_srt_path,
            payload.start_time, payload.end_time
        )

        target_clip["start_time"] = round(payload.start_time, 2)
        target_clip["end_time"] = round(payload.end_time, 2)
        target_clip["subtitle_text"] = payload.subtitle_text.strip()

        meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return ClipMetadata(**target_clip)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal revisi klip: {str(e)}")


@app.get("/files/{job_id}/{filename}")
async def get_generated_file(job_id: str, filename: str):
    safe_filename = Path(filename).name
    file_path = JOBS_DIR / job_id / safe_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File tidak ditemukan.")
    media_type = "video/mp4" if safe_filename.endswith(".mp4") else "application/x-subrip"
    return FileResponse(path=file_path, media_type=media_type, filename=safe_filename)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
