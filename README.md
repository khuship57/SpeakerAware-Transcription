# 🎙️ Speaker Diarization & Transcription Pipeline
**Whisper + VAD + Speaker Clustering**

An **end-to-end Python pipeline** for **speaker diarization** and **speech transcription**, designed to answer the core question:

> **Who spoke, when, and what was said?**

This project processes raw audio files and produces **speaker-labeled transcripts** using modern speech processing techniques such as **Voice Activity Detection (VAD)**, **speaker embedding clustering**, and **OpenAI Whisper** for transcription.

---

## 🚀 Features

### 🔊 Audio Preprocessing
- Audio format conversion & resampling
- Noise reduction & normalization
- Filtering and denoising

### 🎯 Voice Activity Detection (VAD)
- Accurate speech/silence segmentation
- Powered by **SpeechBrain**

### 👥 Speaker Diarization
- Speaker embedding extraction
- Clustering-based speaker segmentation
- Detects **who spoke when**
- Configurable support for overlapping speech

### 📝 Speech Transcription
- High-accuracy transcription using **OpenAI Whisper**
- Supports multiple Whisper model sizes (`tiny` → `large`)

### 🖥️ Interactive UI
- Streamlit-based web interface
- Adjustable preprocessing & diarization parameters
- Real-time transcription preview

---

## 🧠 Use Cases
- 🎧 Meeting & interview transcription
- 📞 Call-center audio analysis
- 🎙️ Podcast & panel discussion processing
- 🧪 Speech research & experimentation
- 📊 Dataset preparation for speech ML models

---

## 🛠️ Tech Stack
- **Python**
- **OpenAI Whisper** (ASR)
- **SpeechBrain** (VAD & embeddings)
- **Clustering-based diarization**
- **Streamlit**
- **Audio signal processing**

---

## 📁 Project Structure

```text
.
├── app.py                     # Streamlit application
├── notebooks/
│   └── speaker_id_pipeline.ipynb   # Development & experiments
├── audio_samples/             # Example input audio files
├── output/                    # Transcripts & diarization output
├── requirements.txt           # Python dependencies
└── README.md
```

## ⚙️ Installation

1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/speaker-diarization-pipeline.git
cd speaker-diarization-pipeline
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
⚠️ Note: GPU acceleration will be used automatically if CUDA is available.

▶️ Running the Application
Launch the Streamlit app:
```bash
streamlit run app.py
```
Then open the provided local URL in your browser.

---

## 🎛️ Configuration Options
- 🎤 Whisper model selection (tiny, base, small, medium, large)
- 🔊 VAD sensitivity control
- 👥 Speaker count range
- ⚡ GPU acceleration (if available)
- 🧩 Modular pipeline for easy extension

## 📤 Example Output
```bash
[00:00:02 - 00:00:06] Speaker 1: Hello everyone, welcome to the meeting.
[00:00:07 - 00:00:12] Speaker 2: Thanks, let’s get started.
```

# Output Includes
- Speaker-labeled transcripts
- Timestamps
- Structured text ready for downstream processing
