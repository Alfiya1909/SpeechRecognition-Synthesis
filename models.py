from django.db import models

# Create your models here.

# Table 1: User Details (For Registered Users)
class UserDetails(models.Model):
    id = models.AutoField(primary_key=True)  # Auto-incrementing integer primary key
    email = models.EmailField(unique=True)
    first_last_name = models.CharField(max_length=255)
    mobile_number = models.CharField(max_length=15, unique=True)
    age = models.IntegerField()
    country = models.CharField(max_length=100)
    zip_code = models.CharField(max_length=10)
    gender = models.CharField(max_length=10, choices=[("Male", "Male"), ("Female", "Female"), ("Other", "Other")])
    password = models.CharField(max_length=255)  # Store hashed passwords

    def __str__(self):
        return self.email

# Table 2: User Activities (For Logged-in Users)
class UserActivity(models.Model):
    user = models.ForeignKey(UserDetails, on_delete=models.CASCADE)  # Link to registered user
    action = models.CharField(max_length=255)  # Example: "Started STT", "Converted Text to Speech"
    transcripted_text = models.TextField(null=True, blank=True)  # Stores STT results & YouTube transcription
    audio_output = models.FileField(upload_to='tts_audio/', null=True, blank=True)  # Stores TTS output
    youtube_video_url = models.URLField(null=True, blank=True)  # Stores YouTube video URL if applicable
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} - {self.action}"

# Table 3: Guest User Activities (For Non-Logged-in Users)
class GuestUserActivity(models.Model):
    session_id = models.CharField(max_length=255)  # Unique session ID for tracking
    action = models.CharField(max_length=255)  # Example: "Used STT", "Converted Text to Speech"
    transcripted_text = models.TextField(null=True, blank=True)  # Stores STT results & YouTube transcription
    audio_output = models.FileField(upload_to='tts_audio/', null=True, blank=True)  # Stores TTS output
    youtube_video_url = models.URLField(null=True, blank=True)  # Stores YouTube video URL if applicable
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Guest - {self.action}"
