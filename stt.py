import app
import whisper
import gradio as gr
import numpy as np
import librosa

# Load medium model for better multilingual support (base is too small)
model = whisper.load_model("base")  # or "medium" for even better results


def transcribe(audio):
    try:
        # Load audio with librosa
        audio_array, sr = librosa.load(audio, sr=16000, mono=True)
        audio_array = audio_array.astype(np.float32)

        # Pad or trim the audio
        audio = whisper.pad_or_trim(audio_array)

        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect language with more reliable method
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        print(f"Detected language: {detected_lang} (confidence: {probs[detected_lang]:.2f})")

        # Use language-specific decoding
        options = whisper.DecodingOptions(
            fp16=False,
            language=detected_lang,  # Use detected language
            without_timestamps=True
        )

        result = whisper.decode(model, mel, options)
        return f"{result.text}\n\n(Detected language: {detected_lang})"

    except Exception as e:
        print(f"Error: {str(e)}")
        return "Error processing audio"


# Improved interface with language support info
gr.Interface(
    title='Multilingual Whisper Transcription',
    description="Supports: English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese +50 more",
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath"),
    outputs=gr.Textbox(label="Transcription", lines=5),
    allow_flagging="never",
).launch()