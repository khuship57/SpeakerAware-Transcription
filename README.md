🎙️ Speaker Diarization & Transcription Pipeline (Whisper + VAD)

An end-to-end Python pipeline for speaker diarization and speech transcription, designed to answer the question:

“Who spoke, when, and what was said?”

This project processes raw audio files and produces speaker-labeled transcripts using modern speech processing techniques such as Voice Activity Detection (VAD), speaker embedding clustering, and OpenAI Whisper for transcription.


✨ Key Features

✅ Audio Preprocessing
Format conversion & resampling

Noise reduction & normalization

Filtering and denoising

✅ Voice Activity Detection (VAD)
Accurate speech / silence segmentation
Powered by SpeechBrain

✅ Speaker Diarization
Speaker embedding extraction
Clustering-based speaker segmentation
Detects who spoke when
Supports overlapping speech (configurable)

✅ Speech Transcription
High-accuracy transcription using OpenAI Whisper
Multiple Whisper model sizes supported

✅ Interactive UI 
Streamlit-based interface
Adjustable preprocessing & diarization parameters
Real-time transcription preview

🧠 Typical Use Cases

🎧 Meeting & interview transcription
📞 Call center audio analysis
🎙️ Podcast & panel discussion processing
🧪 Speech research & experimentation
📊 Dataset preparation for speech ML models

📁 Project Structure
├── app.py                     # Streamlit application
├── notebooks/
│   └── speaker_id_pipeline.ipynb   # Development & experiments
├── audio_samples/             # Example input audio files
├── output/                    # Transcripts & diarization output
├── requirements.txt           # Python dependencies
└── README.md

▶️ Getting Started
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit App
streamlit run app.py


⚙️ Configuration Highlights

🎛 Whisper model selection (tiny → large)
🔊 Adjustable VAD sensitivity
👥 Configurable speaker count range
⚡ GPU acceleration enabled where available
🧩 Modular pipeline for easy extension

📤 Output Example
[00:00:02 - 00:00:06] Speaker 1: Hello everyone, welcome to the meeting.
[00:00:07 - 00:00:12] Speaker 2: Thanks, let’s get started.

Outputs include:
Speaker-labeled transcripts
Timestamps
Structured text ready for downstream processing


🛠️ Tech Stack & Keywords
Python
OpenAI Whisper
SpeechBrain
Speaker Diarization
Voice Activity Detection (VAD)
Audio Processing
ASR (Automatic Speech Recognition)
Clustering-based diarization
Streamlit
