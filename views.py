import html
import json
import os
import time
import PyPDF2
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from docx import Document
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import speech_recognition as sr
import yt_dlp
from pydub import AudioSegment
from gtts import gTTS
import logging
from google.cloud import translate_v2 as translate  # Official Google Translate API client
from SpeechApp.pretrained_models import *
import spacy
from datetime import datetime
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ForCTC
import base64
import io
from .models import UserActivity, UserDetails, GuestUserActivity
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.hashers import make_password, check_password
import uuid
import transformers
import warnings
import logging
from transformers import logging as hf_logging

warnings.simplefilter(action='ignore', category=FutureWarning)
hf_logging.set_verbosity_error()
logging.getLogger().setLevel(logging.WARNING)

transformers.logging.set_verbosity_error()

#Load the trained spaCy model
nlp = spacy.load("trained_model")

#Initialize the Google Translate Client using credentials from settings
translator = translate.Client.from_service_account_json(settings.GOOGLE_TRANSLATE_CREDENTIALS_PATH)

# Load the Wav2Vec2 Model
MODEL_NAME = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, use_safetensors=True)

recognizer = sr.Recognizer()
mic = sr.Microphone()

def register(request):
    if request.method == "POST":
        first_last_name = request.POST["first_last_name"]
        mobile_number = request.POST["mobile_number"]
        email = request.POST["email"]
        age = request.POST["age"]
        country = request.POST["country"]
        zip_code = request.POST["zip_code"]
        gender = request.POST["gender"]
        password = request.POST["password"]
        confirm_password = request.POST["confirm_password"]

        if password != confirm_password:
            messages.error(request, "Passwords do not match!")
            return redirect("register")
        
        if UserDetails.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
            return redirect("register")

        # Save user details with hashed password
        user = UserDetails.objects.create(
            first_last_name=first_last_name,
            mobile_number=mobile_number,
            email=email,
            age=age,
            country=country,
            zip_code=zip_code,
            gender=gender,
            password=make_password(password),
        )
        user.save()

        messages.success(request, "Registration successful! Please log in.")
        return redirect("login")

    return render(request, "register.html")

def user_login(request):
    if request.method == "POST":
        email = request.POST["email"]
        password = request.POST["password"]
        user = UserDetails.objects.filter(email=email).first()

        if user and check_password(password, user.password):
            request.session["user_id"] = user.id

            # Log activity
            log_user_activity(request, "User logged in")

            return redirect("home")  # Redirect to home after login
        else:
            messages.error(request, "Invalid credentials!")
            return redirect("login")

    return render(request, "login.html")

def log_user_activity(request, action, transcripted_text=None, audio_output=None, youtube_video_url=None):
    user_id = request.session.get("user_id")
    
    if user_id:
        try:
            user = UserDetails.objects.get(id=user_id)
            activity = UserActivity(
                user=user,
                action=action,
                transcripted_text=transcripted_text,
                audio_output=audio_output,
                youtube_video_url=youtube_video_url,
            )
            # Handle File Upload Properly
            if audio_output:  # Ensure audio_output is a file and not a string path
                file_name = os.path.basename(audio_output.name)
                activity.audio_output.save(file_name, audio_output)
            
            activity.save()
            return True  # Return success confirmation
        except UserDetails.DoesNotExist:
            return False  # User not found
    return False  # User not logged in

def log_guest_activity(request, action, transcripted_text=None, audio_output=None, youtube_video_url=None):
    # Retrieve or generate a session ID
    session_id = request.session.get("guest_id")
    if not session_id:
        session_id = str(uuid.uuid4())  # Generate unique ID
        request.session["guest_id"] = session_id
        request.session.modified = True  # Ensure session is saved

    # Debugging - Print values before saving
    print(f"Session ID: {session_id}, Action: {action}, Text: {transcripted_text}, Audio: {audio_output}, YouTube: {youtube_video_url}")

    # Save guest activity to database
    try:
        guest_activity = GuestUserActivity.objects.create(
            session_id=session_id,
            action=action,
            transcripted_text=transcripted_text,
            audio_output=audio_output,
            youtube_video_url=youtube_video_url
        )
        print("Guest activity saved successfully:", guest_activity)
        return JsonResponse({"message": "Guest activity logged successfully!"})

    except Exception as e:
        print("Error saving guest activity:", e)
        return JsonResponse({"error": "Failed to save guest activity"}, status=500)

# Intent recognition function
def recognize_intent(transcription):
    intent_mapping = {
        "open home": "navigate_home",
        "go to home": "navigate_home",
        "open speech to text": "navigate_stt",
        "go to speech to text": "navigate_stt",
        "start speech to text": "navigate_stt",
        "open text to speech": "navigate_tts",
        "go to text to speech": "navigate_tts",
        "start text to speech": "navigate_tts",
        "open YouTube transcription": "navigate_yt",
        "go to YouTube transcription": "navigate_yt",
        "start transcription": "start_transcription",
        "stop transcription": "stop_transcription",
        "upload and transcribe": "upload_transcribe",
        "translate this": "translate_text",
        "play the audio": "playback_audio",
        "open login page": "navigate_login",
        "go to open login page": "navigate_login",
        "start login page": "navigate_login",
    }

    # Check if the spoken phrase matches any intent
    for phrase, intent in intent_mapping.items():
        if phrase in transcription.lower():
            return intent
    return None  # No intent matched

# Process voice commands from the home page
def process_voice_command(request):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Speak now.")
        try:
            audio = recognizer.listen(source)
            transcription = recognizer.recognize_google(audio)  # No encoding needed
            print(f"Transcribed Text: {transcription}")

            # Recognize intent
            intent = recognize_intent(transcription)

            return JsonResponse({'transcription': transcription, 'intent': intent})

        except sr.UnknownValueError:
            return JsonResponse({'error': 'Could not understand the audio'}, status=400)
        except sr.RequestError:
            return JsonResponse({'error': 'Speech recognition service unavailable'}, status=500)

def start_voice_recognition(request):
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening... Speak now.")
        audio = recognizer.listen(source)

    # Convert audio to text using Wav2Vec2
    audio_data = torch.tensor(recognizer.recognize_google(audio).encode("utf-8"))
    input_values = processor(audio_data, return_tensors="pt", padding=True).input_values
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0]

    return JsonResponse({"text": text})

def stop_voice_recognition(request):
    return JsonResponse({"status": "Recognition stopped"})

def detect_language(text):
    """Detects the language of the input text using Google Translate API."""
    try:
        result = translator.detect_language(text)  # ✅ Use Google's client method
        detected_lang = result.get("language", None)

        if detected_lang:
            logging.info(f"Detected Language: {detected_lang}")
            return detected_lang
        else:
            logging.error("Language detection failed: No language found in response.")
            return None  # Return None if detection fails
    except Exception as e:
        logging.error(f"Language detection failed: {str(e)}")
        return None  # Return None if an error occurs

def translate_text(text, source_language=None, target_language="en"):
    """Translates text using Google Translate API with an optional source_language parameter."""
    try:
        #If source_language is not provided, detect it automatically
        if not source_language:
            source_language = detect_language(text)
            if not source_language:
                return "Error: Unable to detect source language."

        #If source and target languages are the same, return original text
        if source_language == target_language:
            return text  # No translation needed

        #Use Google Translate API to translate
        result = translator.translate(text, target_language=target_language, source_language=source_language)
        translated_text = result.get("translatedText", "")

        if translated_text:
            return translated_text
        else:
            logging.error("Error: Translation failed.")
            return "Error: Translation failed."
    except Exception as e:
        logging.error(f"Translation failed: {str(e)}")
        return f"Error: Translation failed. {str(e)}"

# Directories for uploads and outputs
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
MEDIA_DIR = os.path.join(settings.BASE_DIR, "media")

# Ensure directories exist
for directory in [UPLOAD_DIR, MEDIA_DIR]:
    os.makedirs(directory, exist_ok=True)

def home(request):
    return render(request, 'home.html')

logger = logging.getLogger(__name__)

@csrf_exempt
def speech_to_text(request):
    if request.method == "POST":
        transcription = request.POST.get("transcription", "").strip()
        language = request.POST.get("language", "en")
        target_language = request.POST.get("target_language", "").strip()

        print(f"Received Transcription: {transcription}")
        print(f"Source Language: {language}, Target Language: {target_language}")

        if not transcription:
            return JsonResponse({"error": "No transcription received"}, status=400)

        translated_text = None
        if target_language and target_language != language:
            try:
                translate_client = translate.Client()
                translation_result = translate_client.translate(
                    transcription, 
                    target_language=target_language
                )
                translated_text = html.unescape(translation_result["translatedText"])
            except Exception as e:
                return JsonResponse({"error": f"Translation failed: {str(e)}"}, status=500)

            except Exception as e:
                print(f"Translation Error: {e}")
                return JsonResponse({"error": f"Translation failed: {str(e)}"}, status=500)
            
            log_user_activity(request, "Converted Speech to Text", transcripted_text=transcription)

            log_guest_activity(request, "Used STT", transcripted_text=transcription)

        response_data = {
            "transcription": transcription,
            "translated_text": translated_text or "No translation requested",
        }

        return JsonResponse(response_data)

    return render(request, "speech_to_text.html")

def extract_text_from_file(file: UploadedFile) -> str:
    # Extracts text from a file (TXT, DOCX, PDF).
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        return file.read().decode("utf-8", errors="ignore")
    
    elif file_extension == 'docx':
        doc = Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    elif file_extension == 'pdf':
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''.join([page.extract_text() for page in pdf_reader.pages])
        return text 
    
    return None  # Unsupported file type

def text_to_speech(request):
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        text_file = request.FILES.get("text_file")
        selected_language = request.POST.get("language", "en")  # Language for the generated speech

        if text_file:
            extracted_text = extract_text_from_file(text_file)
            if extracted_text is None:
                return render(request, "text_to_speech.html", {"error": "Unsupported file format."})
            text = extracted_text.strip()

        if not text:
            return render(request, "text_to_speech.html", {"error": "No text provided for conversion."})

        # Detect the language of the text
        detected_language = detect_language(text)
        logger.info(f"Detected Language: {detected_language}, Target TTS Language: {selected_language}")

        # Force translation if detected language is different from selected language
        if detected_language != selected_language:
            translated_text = translate_text(text, detected_language, selected_language)
        else:
            translated_text = text

        logger.info(f"Final Text for TTS: {translated_text}")

        # Convert translated text to speech
        audio_filename = "output.mp3"
        audio_path = os.path.join(MEDIA_DIR, audio_filename)

        try:
            tts = gTTS(translated_text, lang=selected_language)
            tts.save(audio_path)
            logger.info("TTS conversion successful.")
        except Exception as e:
            logger.error(f"TTS conversion failed: {str(e)}")
            return render(request, "text_to_speech.html", {"error": f"TTS conversion failed: {str(e)}"})
        
        log_user_activity(request, "Converted Text to Speech", transcripted_text=None)

        log_guest_activity(request, "Converted Text to Speech", transcripted_text=None)

        return render(request, "text_to_speech.html", {"audio_file": f"/media/{audio_filename}"})

    return render(request, "text_to_speech.html", {"error": "Please enter text or upload a file."})

# Function to download audio from a YouTube video
def download_audio_from_youtube(video_url):
    """
    Downloads the audio from the given YouTube video URL.
    Saves it in the UPLOAD_DIR folder and returns the file path.
    """
    ydl_opts = {
        'format': 'bestaudio/best',  # Get the best audio quality
        'outtmpl': f'{UPLOAD_DIR}/%(id)s.%(ext)s',  # Save the file with video ID as name
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            audio_path = f"{UPLOAD_DIR}/{info_dict['id']}.mp3"
            logging.info(f"Downloaded audio to: {audio_path}")
            return audio_path
    except Exception as e:
        print(f"Error downloading YouTube audio: {str(e)}")
        return None

# View function to handle YouTube video transcription
def youtube_transcription(request):
    """
    Handles YouTube transcription.
    1. Accepts a YouTube URL from the user.
    2. Downloads the audio from the video.
    3. Transcribes the audio into text.
    4. Detects the language of the transcription.
    """
    if request.method == 'POST':
        video_url = request.POST.get('video_url')

        if not video_url:
            return render(request, 'youtube_transcription.html', {'error': 'Please provide a YouTube URL.'})

        # Step 1: Download YouTube audio
        audio_path = download_audio_from_youtube(video_url)
        if not audio_path:
            return render(request, 'youtube_transcription.html', {'error': 'Failed to download audio from YouTube.'})

        # Step 2: Transcribe the audio
        transcription = transcribe_audio(audio_path)
        
        # Debugging: Print what transcription returns
        logging.info(f"Transcription output: {transcription}")

        transcription = process_youtube_transcription(video_url, download_audio_from_youtube)

        if not transcription:  # Check if transcription is empty or None
            return render(request, 'youtube_transcription.html', {'error': 'Transcription failed or returned empty text.'})
        
        log_guest_activity(request, "YouTube Video Transcribed", transcripted_text=transcription, youtube_video_url=video_url)

        # Step 3: Detect language (Ensure it does not return None)
        detected_language = detect_language(transcription)
        if not detected_language:  # Prevent NoneType error
            detected_language = "unknown"
        
        log_user_activity(request, "YouTube Video Transcribed", transcripted_text=transcription, youtube_video_url=video_url)

        # Step 4: Render results (without translation)
        return render(request, 'youtube_transcription.html', {
            'text': transcription,
            'detected_language': detected_language
        })

    return render(request, 'youtube_transcription.html')

def youtube_transcription(request):
    #Handles YouTube transcription view.
    if request.method == 'POST':
        video_url = request.POST.get('video_url')
        if not video_url:
            return render(request, 'youtube_transcription.html', {'error': 'Please provide a YouTube URL.'})
        
        #Use the transcription function
        transcription = process_youtube_transcription(video_url, download_audio_from_youtube)

        return render(request, 'youtube_transcription.html', {
            'text': transcription
        })
    
    #log_user_activity(request, "YouTube Video Transcribed", transcripted_text=transcription, youtube_video_url=video_url)

    return render(request, 'youtube_transcription.html')

def classify_command(text):
    #Predict the intent of the given text using the trained spaCy model.
    doc = nlp(text)
    return doc.cats #Return category score
