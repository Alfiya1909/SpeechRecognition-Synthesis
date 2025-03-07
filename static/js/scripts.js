function sendToServer(transcript, selectedLanguage = "en", targetLanguage = "en") {
    console.log("📤 Sending to server:", transcript, selectedLanguage, targetLanguage);

    if (!transcript) {
        console.error("❌ Error: No transcription data to send.");
        return;
    }

    fetch("/speech_to_text/", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-CSRFToken": getCSRFToken(),
        },
        body: new URLSearchParams({
            transcription: transcript,
            language: selectedLanguage,
            target_language: targetLanguage,
        }),
    })
    .then(response => response.json())
    .then(data => {
        console.log("✅ Server Response:", data);

        // ✅ Update the UI with the translated text
        let translatedTextElement = document.getElementById("translatedText");
        if (translatedTextElement) {
            translatedTextElement.innerText = data.translated_text || "Translation failed";
        } else {
            console.error("❌ Translated text element not found!");
        }
    })
    .catch(error => console.error("❌ Fetch Error:", error));
}

// Ensure getCSRFToken() works
function getCSRFToken() {
    let cookieValue = null;
    document.cookie.split(";").forEach((cookie) => {
        let trimmedCookie = cookie.trim();
        if (trimmedCookie.startsWith("csrftoken=")) {
            cookieValue = trimmedCookie.substring("csrftoken=".length);
        }
    });
    return cookieValue || "";
}

// ✅ Ensure `sendToServer` is accessible globally
window.sendToServer = sendToServer;

// ✅ Test if function is accessible
console.log("✅ sendToServer is loaded:", typeof window.sendToServer);

document.addEventListener("DOMContentLoaded", function () {
    let isRecording = false;
    let isVoiceCommandsActive = false;
    let recognition, voiceRecognition;

    // Get references to elements
    let startRecordingButton = document.getElementById("startRecording");
    let languageSelect = document.getElementById("languageSelect");
    let targetLanguageSelect = document.getElementById("targetLanguageSelect");
    let transcriptionElement = document.getElementById("transcription");
    let translatedTextElement = document.getElementById("translatedText");
    let voiceCommandToggle = document.getElementById("voice-command-toggle");
    let micIcon = voiceCommandToggle?.querySelector("i");

    // ✅ Speech Recognition (STT)
    if ("webkitSpeechRecognition" in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;

        recognition.onstart = function () {
            isRecording = true;
            console.log("🎤 Recording started...");
            if (startRecordingButton) startRecordingButton.innerText = "Stop Recording";
        };

        recognition.onend = function () {
            isRecording = false;
            console.log("🔴 Recording stopped.");
            if (startRecordingButton) startRecordingButton.innerText = "Start Recording";
        };

        recognition.onerror = function (event) {
            console.error("❌ Recognition error:", event.error);
            if (event.error === "not-allowed") {
                alert("⚠️ Microphone access denied! Please allow microphone access in your browser settings.");
            }
        };

        recognition.onresult = function (event) {
            let transcript = event.results[event.results.length - 1][0].transcript.trim();
            console.log("📜 Transcribed Text:", transcript);

            let selectedLanguage = languageSelect ? languageSelect.value : "en";
            let targetLanguage = targetLanguageSelect ? targetLanguageSelect.value : "en";

            if (transcriptionElement) transcriptionElement.innerText = transcript;

            // Send transcription to the server
            sendToServer(transcript, selectedLanguage, targetLanguage);
        };

        if (startRecordingButton) {
            startRecordingButton.addEventListener("click", function () {
                if (!isRecording) {
                    navigator.mediaDevices.getUserMedia({ audio: true })
                        .then((stream) => {
                            console.log("🎤 Microphone access granted!", stream);
                            
                            // ✅ Store stream to stop it later
                            window.activeStream = stream;
            
                            recognition.start();
                            console.log("🎤 Recording started...");
                        })
                        .catch(err => {
                            console.error("❌ Microphone access error:", err);
                            alert("⚠️ Please allow microphone access in your browser settings.");
                        });
                } else {
                    recognition.stop();
            
                    // ✅ Stop the microphone stream properly
                    if (window.activeStream) {
                        let tracks = window.activeStream.getTracks();
                        tracks.forEach(track => track.stop()); // Stop each track
                        console.log("🎤 Microphone access released.");
                    }
                }
            });                      
            
        } else {
            console.error("❌ Error: Start Recording button not found.");
        }
    } else {
        alert("⚠️ Your browser does not support speech recognition. Please use Google Chrome.");
    }

    // ✅ Voice Command Recognition
    if ("webkitSpeechRecognition" in window) {
        voiceRecognition = new webkitSpeechRecognition();
        voiceRecognition.continuous = true;
        voiceRecognition.interimResults = false;
        voiceRecognition.lang = "en-US";

        voiceRecognition.onresult = function (event) {
            let command = event.results[event.results.length - 1][0].transcript.trim().toLowerCase();
            console.log("🎙️ Voice Command Detected:", command);
            processVoiceCommand(command);
        };

        voiceRecognition.onerror = function (event) {
            console.error("❌ Voice command error:", event.error);
            stopVoiceCommands();
        };

        voiceRecognition.onend = function () {
            if (isVoiceCommandsActive) {
                console.log("🔄 Restarting voice command recognition...");
                setTimeout(() => voiceRecognition.start(), 1000);
            }
        };

        function startVoiceCommands() {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(() => {
                    isVoiceCommandsActive = true;
                    voiceRecognition.start();
                    micIcon?.classList.add("text-danger");
                    console.log("🎤 Voice Commands Activated");
                })
                .catch(err => {
                    console.error("❌ Microphone access error:", err);
                    alert("⚠️ Please allow microphone access in your browser settings.");
                });
        }

        function stopVoiceCommands() {
            isVoiceCommandsActive = false;
            voiceRecognition.stop();
            micIcon?.classList.remove("text-danger");
            console.log("🔇 Voice Commands Deactivated");
        }

        voiceCommandToggle?.addEventListener("click", function () {
            isVoiceCommandsActive ? stopVoiceCommands() : startVoiceCommands();
        });

        function processVoiceCommand(command) {
            let commandMap = {
                "open speech to text": "/speech_to_text/",
                "go to speech to text": "/speech_to_text/",
                "open text to speech": "/text_to_speech/",
                "go to text to speech": "/text_to_speech/",
                "open youtube transcription": "/youtube_transcription/",
                "go to youtube transcription": "/youtube_transcription/",
                "start": () => document.getElementById("startRecording")?.click(),
            };

            if (command in commandMap) {
                console.log(`✅ Executing command: ${command}`);
                let action = commandMap[command];
                if (typeof action === "string") {
                    window.location.href = action;
                } else if (typeof action === "function") {
                    action();
                }
            } else {
                console.log(`⚠️ Unrecognized command: "${command}"`);
            }
        }
    }

});


// Convert Text to Speech with Translation
document.getElementById('convertToSpeech').addEventListener('click', function() {
    let text = document.getElementById('textInput').value;
    let targetLanguage = document.getElementById('ttsLanguageSelect').value;

    if (text !== '') {
        fetch('/text_to_speech/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': getCSRFToken()
            },
            body: `text=${encodeURIComponent(text)}&language=${targetLanguage}`
        })
        .then(response => response.json())
        .then(data => {
            if (data.audio_url) {
                let audio = new Audio(data.audio_url);
                audio.play();
            } else {
                alert('TTS conversion failed');
            }
        })
        .catch(error => {
            console.error('Error during TTS conversion:', error);
        });
    } else {
        alert('Please enter text to convert to speech.');
    }
});

// Get CSRF Token Function
function getCSRFToken() {
    let cookieValue = null;
    if (document.cookie) {
        let cookies = document.cookie.split(';');
        cookies.forEach(cookie => {
            let trimmedCookie = cookie.trim();
            if (trimmedCookie.startsWith('csrftoken=')) {
                cookieValue = trimmedCookie.substring('csrftoken='.length);
            }
        });
    }
    return cookieValue;
}
