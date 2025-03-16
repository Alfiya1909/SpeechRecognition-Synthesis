from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('speech_to_text/', views.speech_to_text, name='speech_to_text'),
    path('text_to_speech/', views.text_to_speech, name='text_to_speech'),
    path('youtube_transcription/', views.youtube_transcription, name='youtube_transcription'),
    path('translate_text/', views.translate_text, name='translate_text'),
    path("start-voice-recognition/", views.start_voice_recognition, name="start_voice_recognition"),
    path("stop-voice-recognition/", views.stop_voice_recognition, name="stop_voice_recognition"),
    path("process-voice-command/", views.process_voice_command, name="process_voice_command"),
    path("register/", views.register, name="register"),
    path("login/", views.user_login, name="login"),

]

# Serve media files in development mode
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
