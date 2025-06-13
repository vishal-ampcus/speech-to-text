import whisper
import gradio as gr
import numpy as np
import librosa
import noisereduce as nr


model = whisper.load_model("medium")


def transcribe(audio_path):
    try:
        if audio_path is None:
            return "⚠️ No audio file received. Please try again."

        print(f"Audio file received: {audio_path}")

        # Load the audio using librosa
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Normalize
        audio_array = audio_array / np.max(np.abs(audio_array) + 1e-8)

        # Optional: apply noise reduction
        audio_array = nr.reduce_noise(y=audio_array, sr=sr)

        # Whisper expects 30 seconds of audio. Pad or trim to that
        audio = whisper.pad_or_trim(audio_array)

        # Convert to Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect language
        _, probs = model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        print(f"Detected language: {detected_lang} ({probs[detected_lang]:.2f})")

        # Decode
        options = whisper.DecodingOptions(language=detected_lang, fp16=False, without_timestamps=True)
        result = whisper.decode(model, mel, options)

        return f"{result.text}\n\n(Detected Language: {detected_lang})"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error processing audio:\n{e}"


# Gradio Interface
gr.Interface(
    title="Speech To Text",
    description="Upload or record audio",
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎙️ Input Audio"),
    outputs=gr.Textbox(label="📝 Transcription", lines=10),
    allow_flagging="never"
).launch()
