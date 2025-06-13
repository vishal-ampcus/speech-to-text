from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import whisper
import tempfile
import os
from typing import Optional

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model at startup (choose appropriate size)
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    language: Optional[str] = None,
    task: Optional[str] = "transcribe"  # or "translate"
):
    temp_path = None
    try:
        # Validate file type
        valid_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.mp4'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in valid_extensions:
            raise HTTPException(400, detail=f"Unsupported file format. Supported: {valid_extensions}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Transcribe with Whisper
        result = model.transcribe(
            temp_path,
            language=language,  # None for auto-detection
            task=task,  # "transcribe" or "translate"
            fp16=False  # Set to True if using GPU
        )

        return {
            "text": result["text"],
            "language": result["language"],
            "segments": result.get("segments", []),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)