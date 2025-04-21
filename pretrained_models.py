import os
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, SpeechT5ForTextToSpeech, SpeechT5Processor
from spacy import load
import logging
import soundfile as sf
from textblob import TextBlob
from google.cloud import translate_v2 as translate
import whisper
from pydub import AudioSegment
import scipy.io.wavfile as wav
import noisereduce as nr
import spacy
from spacy.util import minibatch, compounding
import random
from spacy.training.example import Example  # ✅ Import Example class

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load NLP model (spaCy)
nlp = load("en_core_web_sm")

# Load device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load Wav2Vec2 model for speech-to-text
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# Load Hugging Face model for text-to-speech
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

#Intailize the Google Translate client
translate_client = translate.Client

#Load Whisper model once for efficiency
WHISPER_MODEL = whisper.load_model("base")

def transcribe_audio(audio_path):
    """Transcribe audio to text with proper formatting."""
    try:
        logging.info(f"Starting transcription for: {audio_path}")

        # Load audio
        speech, rate = librosa.load(audio_path, sr=16000)
        if len(speech) == 0:
            logging.error("Loaded audio is empty.")
            return None
        
        speech = np.array(speech, dtype=np.float32)

        # Ensure model processing
        input_values = processor(speech, sampling_rate=rate, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # ✅ Fix transcription formatting
        transcription = transcription.strip().replace("\n", " ")

        # ✅ Avoid printing unnecessary debug info
        logging.info(f"Final Transcription: {transcription}")

        return transcription
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return None

def text_to_speech(text, output_path):
    """Convert text to speech using Hugging Face SpeechT5.
    Args:
        text (str): Input text to be converted to speech.
        output_path (str): Path to save the generated audio file.
    Returns:
        str: File path of the generated audio.
    """
    logging.info(f"Converting text to speech: {text}")
    try:
        if not text.strip():
            logging.error("Empty text provided for TTS.")
            return
        
        inputs = tts_processor(text, return_tensors="pt").to(device)
        speaker_embeddings = torch.zeros((1, 512), device=device)  # Placeholder for embeddings
        speech = tts_model.generate_speech(inputs["input_ids"], speaker_embeddings.to(device))
        sf.write(output_path, speech.cpu().numpy(), 16000)
        logging.info(f"Speech saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error during text-to-speech conversion: {str(e)}")

def get_translation_model(source_lang, target_lang):
    """Dynamically load the correct translation model."""
    if source_lang == target_lang:
        logging.warning("Source and target languages are the same. No translation needed.")
        return None
    
    logging.info(f"Loading Google Translate model for {source_lang} to {target_lang}")
    try:
        return translate_client, source_lang, target_lang
    except Exception as e:
        logging.error(f"Error loading Google Translate client: {str(e)}")
        return None

def translate_text(text, source_lang, target_lang):
    try:
        client, src_lang, tgt_lang = get_translation_model(source_lang, target_lang)

        if not client:
            return "Translation model unavailable for this language pair"
        
        #Perform the translation using the Google Translate API
        translation = client.translate(text, source_language=src_lang, target_language=tgt_lang)

        #Return the translated text
        translated_text = translation['translatedText']
        return translated_text
    except Exception as e:
        logging.error(f"Translation failed: {str(e)}")
        return "Translation failed"

def reduce_noise(input_path, output_path, chunk_size=50000):
    """Reduce noise from an audio file in chunks to avoid MemoryError."""
    audio = AudioSegment.from_file(input_path)
    
    # Convert to WAV format
    audio.export(output_path, format="wav")

    # Read WAV file
    rate, data = wav.read(output_path)

    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1).astype(np.int16)

    # Process noise reduction in chunks
    reduced_noise = np.zeros_like(data)
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        reduced_chunk = nr.reduce_noise(y=chunk, sr=rate, prop_decrease=0.8)
        reduced_noise[i : i + len(reduced_chunk)] = reduced_chunk

    # Save the cleaned audio
    wav.write(output_path, rate, reduced_noise.astype(np.int16))

    return output_path

def transcribe_audio_whisper(audio_path):
    #Transcribe audio using local Whipser model.
    result = WHISPER_MODEL.transcribe(audio_path)
    return result["text"]

def correct_transcription(text):
    #Fix typos in the transcription using TextBlob.
    return str(TextBlob(text).correct())

def process_youtube_transcription(video_url, download_audio_from_youtube):
    """Complete pipeline for transcribing YouTube videos."""
    audio_path = download_audio_from_youtube(video_url)
    
    if not audio_path:
        return "Error: Failed to download audio."

    # Define an output path for the cleaned audio
    output_audio_path = audio_path.rsplit(".", 1)[0] + "_clean.wav"

    # Fix function call: Pass both input and output paths
    cleaned_audio = reduce_noise(audio_path, output_audio_path)
    
    raw_transcription = transcribe_audio_whisper(cleaned_audio)  # Local Whisper transcription
    corrected_text = correct_transcription(raw_transcription)  # Correct typos

    return corrected_text  # Final refined output

def get_sentiment(text):
    """Get the sentiment of the text."""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def test_transcription(audio_path):
    try:
        transcription_data = transcribe_audio(audio_path)
        if transcription_data:
            logging.info(f"Transcription: {transcription_data['transcription']}")
    except Exception as e:
        logging.error(f"Error during transcription test: {str(e)}")

# Define intent-action mapping
COMMANDS = {
    "open speech to text": "navigate_stt",
    "open text to speech": "navigate_tts",
    "open youtube transcription": "navigate_yt",
    "start": "start_transcription",
    "stop": "stop_transcription",
    "upload the file and convert to text": "upload_transcribe",
    "translate text": "translate_text",
    "playback audio": "playback_audio",
    "open login page": "navigate_login"
}

def recognize_intent(text):
    """Process text and identify intent."""
    doc = nlp(text.lower())
    
    for command in COMMANDS:
        if command in doc.text:
            return COMMANDS[command]

    return None  # No matching command
