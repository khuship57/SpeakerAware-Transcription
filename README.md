# 🎙️ Speaker Identification and Transcription Pipeline

This project performs end-to-end audio processing, including:

- Audio preprocessing (conversion, filtering, normalization, denoising)
- Voice Activity Detection (VAD) using SpeechBrain
- Speaker Embedding extraction and clustering
- Speaker Diarization (who spoke when)
- Transcription using OpenAI's Whisper
- Optional Streamlit interface for interaction

## 📁 Project Structure
```
├── app.py                   # Streamlit app
├── notebooks/
│   └── speaker_id_pipeline.ipynb  # Development notebook
├── audio_samples/          # Sample input audio (add your own)
├── output/                 # Output files (ignored in .gitignore)
├── models/                 # Pretrained models (ignored)
├── tests/                  # Test scripts (optional)
├── requirements.txt        # Dependencies
├── .gitignore              # Ignored files/folders
└── README.md
```

## ▶️ Run the App
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📌 Notes
- Whisper model selection is available in the sidebar.
- All processing is GPU-accelerated where possible.
- Preprocessing options and speaker range are customizable.