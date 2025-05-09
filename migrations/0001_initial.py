# Generated by Django 5.1.4 on 2025-03-09 17:20

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='GuestUserActivity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_id', models.CharField(max_length=255)),
                ('action', models.CharField(max_length=255)),
                ('transcripted_text', models.TextField(blank=True, null=True)),
                ('translated_text', models.TextField(blank=True, null=True)),
                ('audio_output', models.FileField(blank=True, null=True, upload_to='tts_audio/')),
                ('youtube_video_url', models.URLField(blank=True, null=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='UserDetails',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('first_last_name', models.CharField(max_length=255)),
                ('mobile_number', models.CharField(max_length=15, unique=True)),
                ('age', models.IntegerField()),
                ('country', models.CharField(max_length=100)),
                ('zip_code', models.CharField(max_length=10)),
                ('gender', models.CharField(choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], max_length=10)),
                ('password', models.CharField(max_length=255)),
            ],
        ),
        migrations.CreateModel(
            name='UserActivity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('action', models.CharField(max_length=255)),
                ('transcripted_text', models.TextField(blank=True, null=True)),
                ('translated_text', models.TextField(blank=True, null=True)),
                ('audio_output', models.FileField(blank=True, null=True, upload_to='tts_audio/')),
                ('youtube_video_url', models.URLField(blank=True, null=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='SpeechApp.userdetails')),
            ],
        ),
    ]
