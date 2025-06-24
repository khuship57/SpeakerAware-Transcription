import streamlit as st
import os
import tempfile
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import re
import pickle
from threading import Thread
from pyngrok import ngrok
import streamlit as st

# Audio processing libraries
from pydub import AudioSegment
import librosa
import soundfile as sf
import scipy.signal
import torch
import torchaudio
import whisper
from datetime import timedelta

# Machine learning libraries
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# SpeechBrain imports (make sure these are installed)
try:
    from speechbrain.pretrained import VAD, EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    st.error("SpeechBrain is not installed. Please install it using: pip install speechbrain")

warnings.filterwarnings("ignore")

def check_ffmpeg_installation():
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ffmpeg_apt():
    try:
        st.info("🔄 Installing FFmpeg using apt...")
        subprocess.run(['apt', 'update'], check=True, capture_output=True)
        subprocess.run(['apt', 'install', '-y', 'ffmpeg'], check=True, capture_output=True)
        return True

    except subprocess.CalledProcessError as e:
        st.error(f"❌ Failed to install FFmpeg: {e}")
        return False
    except FileNotFoundError:
        st.error("❌ apt command not found. This system may not support apt package manager.")
        return False
        
def setup_ffmpeg():
    """Setup FFmpeg for the environment"""
    if check_ffmpeg_installation():
        return True
    
    st.warning("⚠️ FFmpeg not found. Attempting to install...")
    
    # Try different installation methods
    if install_ffmpeg_apt():
        return check_ffmpeg_installation()
    
    # Alternative: Try to set AudioSegment to use system libraries
    st.info("🔄 Trying alternative audio conversion methods...")
    try:
        # Try to use AudioSegment with different converters
        AudioSegment.converter = "ffmpeg"
        AudioSegment.ffmpeg = "ffmpeg"
        AudioSegment.ffprobe = "ffprobe"
        return True
    except:
        pass
    
    st.error("""
    ❌ Could not install or configure FFmpeg. Please install it manually:
    
    **For Ubuntu/Debian:**
    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```
    
    **For other systems:**
    - Visit: https://ffmpeg.org/download.html
    - Or use your system's package manager
    """)
    return False

# Configure page
st.set_page_config(
    page_title="VoiceShield",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #003344;
        margin: 1rem 0;
    }
    .help-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .tip-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        color: #003344;
        margin: 1rem 0;
    }
    .info-box h4,
    .tip-box h4,
    .success-box h4,
    .help-box h4 {
        color: #1a1a1a !important;
    }

    /* Ensure all text in custom boxes is readable */
    .info-box p,
    .tip-box p,
    .success-box p,
    .help-box p,
    .info-box li,
    .tip-box li,
    .success-box li,
    .help-box li {
        color: #2c2c2c !important;
    }

    /* Override Streamlit's default heading colors */
    .info-box h1,
    .info-box h2,
    .info-box h3,
    .info-box h4,
    .info-box h5,
    .info-box h6,
    .tip-box h1,
    .tip-box h2,
    .tip-box h3,
    .tip-box h4,
    .tip-box h5,
    .tip-box h6,
    .success-box h1,
    .success-box h2,
    .success-box h3,
    .success-box h4,
    .success-box h5,
    .success-box h6,
    .help-box h1,
    .help-box h2,
    .help-box h3,
    .help-box h4,
    .help-box h5,
    .help-box h6 {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

class AudioProcessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        # Set device for audio processing (can use GPU for librosa operations)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def convert_to_wav(self, input_file, output_path):
        """Convert audio file to WAV format with improved error handling"""
        try:
            # Method 1: Try using pydub with FFmpeg
            try:
                audio = AudioSegment.from_file(input_file)
                audio.export(output_path, format="wav")
                return True, "Conversion successful using pydub"
            except Exception as e1:
                st.warning(f"Pydub conversion failed: {str(e1)}")
                
                # Method 2: Try using librosa directly
                try:
                    audio, sr = librosa.load(str(input_file), sr=None)
                    sf.write(output_path, audio, sr)
                    return True, "Conversion successful using librosa"
                except Exception as e2:
                    st.warning(f"Librosa conversion failed: {str(e2)}")
                    
                    # Method 3: Try using torchaudio
                    try:
                        waveform, sample_rate = torchaudio.load(str(input_file))
                        # Convert to mono if stereo
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        torchaudio.save(output_path, waveform, sample_rate)
                        return True, "Conversion successful using torchaudio"
                    except Exception as e3:
                        return False, f"All conversion methods failed: pydub({e1}), librosa({e2}), torchaudio({e3})"
                        
        except Exception as e:
            return False, f"Conversion failed: {str(e)}"

    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        """Load audio from a file."""
        try:
            audio, sr = librosa.load(path, sr=None)
        except Exception:
            audio, sr = sf.read(path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
        return audio, sr

    def resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resample audio to the target sample rate."""
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        return audio

    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio signal using peak method only."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = 0.95 * audio / peak
        return np.clip(audio, -1.0, 1.0)

    def highpass_filter(self, audio: np.ndarray, cutoff: float = 80.0) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise."""
        nyquist = self.target_sr / 2
        b, a = scipy.signal.butter(4, cutoff / nyquist, btype='high')
        return scipy.signal.filtfilt(b, a, audio)

    def spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Reduce noise via spectral subtraction."""
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag, phase = np.abs(stft), np.angle(stft)
        noise_est = np.mean(mag[:, :min(10, mag.shape[1] // 4)], axis=1, keepdims=True)
        cleaned_mag = np.maximum(mag - 2.0 * noise_est, 0.1 * mag)
        enhanced = librosa.istft(cleaned_mag * np.exp(1j * phase), hop_length=512)
        return enhanced

    def preprocess_audio(self, audio: np.ndarray, sr: int, **kwargs) -> np.ndarray:
        """Apply full preprocessing pipeline to audio."""
        audio = self.resample(audio, sr)

        if kwargs.get('apply_highpass', True):
            audio = self.highpass_filter(audio)
        if kwargs.get('denoise_method') == 'spectral_subtraction':
            audio = self.spectral_subtraction(audio)

        audio = self.normalize(audio) 
        return audio

class SpeakerDiarization:
    def __init__(self):
        if not SPEECHBRAIN_AVAILABLE:
            st.error("SpeechBrain is required for speaker diarization")
            return

        # Force VAD to use CPU to avoid cuDNN GRU issues
        self.vad_device = "cpu"
        # Use GPU for speaker embedding extraction if available
        self.speaker_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000

        print(f"VAD will run on: {self.vad_device}")
        print(f"Speaker embedding will run on: {self.speaker_device}")

    def load_vad_model(self):
        """Load Voice Activity Detection model - FORCED TO CPU"""
        try:
            # Explicitly force VAD model to CPU
            self.vad = VAD.from_hparams(
                source="speechbrain/vad-crdnn-libriparty",
                run_opts={"device": self.vad_device}  # Always CPU for VAD
            )

            # Ensure VAD model parameters are on CPU
            for param in self.vad.parameters():
                param.data = param.data.cpu()

            print("✅ VAD model loaded successfully on CPU")
            return True
        except Exception as e:
            st.error(f"Failed to load VAD model: {str(e)}")
            print(f"VAD model error: {str(e)}")
            return False

    def load_speaker_model(self):
        """Load Speaker Recognition model - CAN USE GPU"""
        try:
            # Speaker model can use GPU if available
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.speaker_device}
            )
            print(f"✅ Speaker model loaded successfully on {self.speaker_device}")
            return True
        except Exception as e:
            st.error(f"Failed to load speaker model: {str(e)}")
            print(f"Speaker model error: {str(e)}")
            return False

    def get_speech_segments(self, audio_path, threshold=0.5, min_duration=0.25):
        """Detect speech segments using VAD - RUNS ON CPU ONLY"""
        try:
            print("🔄 Starting VAD processing on CPU...")

            # Clear any GPU cache first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            waveform = waveform.float().contiguous()

            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.sample_rate
                ).float().contiguous()

            # CRITICAL: Keep everything on CPU for VAD
            signal = waveform[0].cpu().float().contiguous()

            with torch.no_grad():
                # Ensure input tensor is on CPU and contiguous
                input_tensor = signal.unsqueeze(0).cpu().float().contiguous()
                input_tensor = input_tensor.to(memory_format=torch.contiguous_format)

                print(f"VAD input tensor device: {input_tensor.device}")
                print(f"VAD input tensor shape: {input_tensor.shape}")

                # Run VAD inference on CPU
                probs = self.vad.forward(input_tensor)
                print("✅ VAD inference completed successfully")

            # Process results (all on CPU)
            probs = probs.squeeze().cpu().numpy()
            frame_duration = 0.01
            segments = []
            start = None

            for i, p in enumerate(probs):
                t = i * frame_duration
                if p > threshold:
                    if start is None:
                        start = t
                else:
                    if start is not None:
                        end = t
                        if end - start >= min_duration:
                            segments.append((start, end))
                        start = None

            # Handle final segment
            if start is not None:
                end = len(probs) * frame_duration
                if end - start >= min_duration:
                    segments.append((start, end))

            print(f"✅ VAD completed: found {len(segments)} speech segments")
            return segments

        except Exception as e:
            print(f"❌ Error in VAD processing: {e}")
            import traceback
            traceback.print_exc()

            # Clean up GPU memory if error occurs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def extract_speaker_embedding(self, audio_path):
        """Extract speaker embedding - CAN USE GPU"""
        try:
            print(f"🔄 Extracting speaker embedding on {self.speaker_device}...")

            # Load audio
            signal, fs = torchaudio.load(audio_path)

            # Check minimum length
            if signal.shape[1] < 16000:
                print("⚠️ Audio segment too short for embedding")
                return None

            # Resample if needed
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(fs, 16000)
                signal = resampler(signal)

            # Convert to mono if needed
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)

            # Move to appropriate device for speaker model
            signal = signal.to(self.speaker_device)

            with torch.no_grad():
                # Extract embedding using GPU if available
                embedding = self.classifier.encode_batch(signal)
                embedding = embedding.squeeze().cpu().numpy()

            # Validate embedding
            if embedding.size == 0 or np.any(np.isnan(embedding)):
                print("⚠️ Invalid embedding extracted")
                return None

            print("✅ Speaker embedding extracted successfully")
            return embedding

        except Exception as e:
            print(f"❌ Error extracting embedding: {str(e)}")
            return None

def main():
    st.title("🎙️ Smart Audio Analyzer")
    st.markdown("Turn your recordings into structured insights with intelligent speaker detection and transcription.")

    # Main information box about the application features
    st.markdown("""
    <div class="info-box">
        <h4>🎯 What This App Does</h4>
        <p>This tool analyzes your audio recordings and provides:</p>
        <ul>
            <li><strong>Speaker Diarization:</strong> Identifies and separates different speakers in your recording</li>
            <li><strong>Transcription:</strong> Converts speech to text with speaker labels and timestamps</li>
            <li><strong>Audio Enhancement:</strong> Cleans up audio quality for better processing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("🛠️ Configuration")

    # Recording Type Selection with Radio Buttons
    st.sidebar.markdown("### 🎯 Recording Type")
    recording_type = st.sidebar.radio(
        "Select the type of recording you have:",
        options=["Single Speaker", "Multiple Speakers"],
        index=1,  # Default to Single Speaker
        help="Choose based on whether your recording contains one person or multiple people speaking"
    )

    # Determine if single speaker based on radio selection
    is_single_speaker = (recording_type == "Single Speaker")

    # Display status message based on selection
    if is_single_speaker:
        st.sidebar.success("✅ Single-speaker mode: Fast & accurate for solo recordings")
    else:
        st.sidebar.info("🎭 Multi-speaker mode: Will identify and separate different voices")

    # Conditional speaker diarization options - only show for Multiple Speakers
    min_speakers = 2
    max_speakers = 8

    if not is_single_speaker:
        st.sidebar.markdown("### 👥 Speaker Detection Settings")
        st.sidebar.markdown("*Set the expected range of speakers in your recording*")

        min_speakers = st.sidebar.number_input(
            "Minimum Speakers",
            min_value=1,
            max_value=10,
            value=2,
            help="The minimum number of different speakers you expect"
        )
        max_speakers = st.sidebar.number_input(
            "Maximum Speakers",
            min_value=1,
            max_value=10,
            value=8,
            help="The maximum number of different speakers you expect"
        )

        if min_speakers > max_speakers:
            st.sidebar.error("⚠️ Minimum speakers cannot be greater than maximum speakers")

    st.sidebar.markdown("### 🤖 AI Model Settings")
    whisper_model_name = st.sidebar.selectbox(
        "Whisper Transcription Model",
        options=["tiny", "base", "small", "medium", "large"],
        index=2,  # default to "small"
        help="Larger models are more accurate but slower. 'small' is recommended for most use cases."
    )

    # Model performance guide
    model_info = {
        "tiny": "⚡ Fastest, basic accuracy",
        "base": "🏃 Fast, good accuracy",
        "small": "⚖️ Balanced speed/accuracy",
        "medium": "🎯 High accuracy, slower",
        "large": "🏆 Best accuracy, slowest"
    }
    st.sidebar.caption(f"**{whisper_model_name.title()} Model:** {model_info[whisper_model_name]}")

    # File format support info
    st.sidebar.markdown("### 📄 Supported Formats")
    st.sidebar.caption("✅ WAV, MP3, FLAC, M4A, OGG")
    st.sidebar.caption("💡 WAV files typically provide best results")

    # File upload
    st.header("📁 Upload Your Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file to analyze",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        help="Drag and drop your audio file here, or click to browse. Maximum file size depends on your system memory."
    )

    # File quality tips
    if not uploaded_file:
        st.markdown("""
        <div class="tip-box">
            <h4>📋 Tips for Best Results</h4>
            <ul>
                <li><strong>Audio Quality:</strong> Clear recordings with minimal background noise work best</li>
                <li><strong>File Format:</strong> WAV files typically give the most accurate results</li>
                <li><strong>Length:</strong> Works with recordings from a few seconds to several hours</li>
                <li><strong>Language:</strong> Optimized for English, but supports many languages</li>
                <li><strong>Multiple Speakers:</strong> Ensure speakers are clearly distinguishable and don't overlap too much</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save uploaded file
            input_file = temp_path / uploaded_file.name
            with open(input_file, "wb") as f:
                f.write(uploaded_file.read())

            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

            # Display file info
            file_stats = os.stat(input_file)
            file_size_mb = file_stats.st_size / (1024*1024)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            with col2:
                st.metric("Format", uploaded_file.name.split('.')[-1].upper())

            # Audio preview
            st.subheader("🎵 Audio Preview")
            st.audio(str(input_file))

            # Display processing mode information
            if is_single_speaker:
                st.markdown("""
                <div class="success-box">
                    <h4>🎯 Single Speaker Mode Selected</h4>
                    <p>Your audio will be processed as a single-speaker recording:</p>
                    <ul>
                        <li>✅ Direct transcription (faster processing)</li>
                        <li>✅ Higher accuracy for solo content</li>
                        <li>✅ Complete transcript with timestamps</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                    <h4>🎭 Multi-Speaker Mode Selected</h4>
                    <p>Your audio will be analyzed for multiple speakers:</p>
                    <ul>
                        <li>🔍 Will detect {min_speakers}-{max_speakers} different speakers</li>
                        <li>🏷️ Each speaker will be labeled (Speaker_0, Speaker_1, etc.)</li>
                        <li>📊 Includes speaker distribution analysis</li>
                        <li>⏱️ Processing may take longer for thorough analysis</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Processing warning for large files
            if file_size_mb > 50:
                st.warning("⚠️ Large file detected. Processing may take several minutes. Please be patient.")

            if st.button("🚀 Start Audio Analysis", type="primary", use_container_width=True):
                with st.spinner("Processing your audio... This may take a few minutes."):
                    process_audio_pipeline(
                        input_file=input_file,
                        temp_path=temp_path,
                        apply_highpass=True,
                        apply_denoising=True,
                        is_single_speaker=is_single_speaker,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        whisper_model_name=whisper_model_name
                    )

def process_audio_pipeline(input_file, temp_path, apply_highpass,
                          apply_denoising, is_single_speaker, min_speakers, max_speakers, whisper_model_name):
    """Main processing pipeline with optimized GPU/CPU usage"""

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Convert to WAV
        status_text.text("🔄 Converting to WAV format...")
        processor = AudioProcessor()
        wav_file = temp_path / "converted.wav"

        success, message = processor.convert_to_wav(input_file, wav_file)
        if not success:
            st.error(f"❌ Conversion failed: {message}")
            return

        progress_bar.progress(10)

        # Step 2: Preprocess audio
        status_text.text("🔄 Preprocessing audio (noise reduction, normalization)...")
        audio, sr = processor.load_audio(str(wav_file))

        processed_audio = processor.preprocess_audio(
            audio, sr,
            normalize_method=normalize_method,
            apply_highpass=apply_highpass,
            denoise_method='spectral_subtraction' if apply_denoising else None
        )

        preprocessed_file = temp_path / "preprocessed.wav"
        sf.write(preprocessed_file, processed_audio, processor.target_sr)
        progress_bar.progress(25)

        # Branch based on single speaker mode
        if is_single_speaker:
            # Single speaker mode: Direct transcription
            status_text.text("🔄 Transcribing single speaker audio with AI...")
            progress_bar.progress(50)

            transcripts = transcribe_single_speaker(
                str(preprocessed_file), temp_path, whisper_model_name
            )
            progress_bar.progress(100)

            # Display results for single speaker
            status_text.text("✅ Processing complete!")
            display_single_speaker_results(transcripts)

        else:
            # Multi-speaker mode: Full diarization pipeline

            # Step 3: Voice Activity Detection (CPU ONLY)
            if not SPEECHBRAIN_AVAILABLE:
                st.error("❌ SpeechBrain not available. Cannot proceed with speaker diarization.")
                return

            status_text.text("🔄 Detecting speech segments (finding where people talk)...")
            diarizer = SpeakerDiarization()

            if not diarizer.load_vad_model():
                st.error("❌ Failed to load voice activity detection model")
                return

            speech_segments = diarizer.get_speech_segments(str(preprocessed_file))
            st.info(f"🎯 Found {len(speech_segments)} speech segments in your recording")
            progress_bar.progress(40)

            # Step 4: Create overlapping segments
            status_text.text("🔄 Creating audio segments for analysis...")
            segments_dir = temp_path / "segments"
            segments_dir.mkdir(exist_ok=True)

            segment_files = create_overlapping_segments(
                str(preprocessed_file), segments_dir, speech_segments
            )
            progress_bar.progress(55)

            # Step 5: Extract speaker embeddings (GPU if available)
            status_text.text(f"🔄 Analyzing speaker voices (AI identification)...")

            if not diarizer.load_speaker_model():
                st.error("❌ Failed to load speaker identification model")
                return

            embeddings, filenames = extract_embeddings(segment_files, diarizer)
            if len(embeddings) == 0:
                st.error("❌ No valid speaker patterns found in audio")
                return

            progress_bar.progress(70)

            # Step 6: Speaker clustering (CPU)
            status_text.text("🔄 Grouping speech by different speakers...")
            speaker_labels = cluster_speakers(
                embeddings, filenames, min_speakers, max_speakers
            )
            progress_bar.progress(85)

            # Step 7: Transcription (GPU if available for Whisper)
            status_text.text(f"🔄 Converting speech to text with speaker labels...")
            transcripts = transcribe_segments(
                str(preprocessed_file), speaker_labels, temp_path, whisper_model_name
            )
            progress_bar.progress(100)

            # Display results for multi-speaker
            status_text.text("✅ Processing complete!")
            display_results(transcripts, speaker_labels)

    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.exception(e)

        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def transcribe_single_speaker(audio_file, temp_path, whisper_model_name):
    """Transcribe audio directly for single speaker"""
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print(f"Loading Whisper model on {device}...")
        model = whisper.load_model(whisper_model_name, device=device)
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        return []

    try:
        # Transcribe the entire audio file
        print("Transcribing single speaker audio...")
        result = model.transcribe(audio_file, task="translate")

        # Get audio duration for metadata
        y, sr = librosa.load(audio_file, sr=16000)
        duration = len(y) / sr

        transcript = [{
            'speaker': 'Speaker_0',
            'start': 0.0,
            'end': duration,
            'text': result['text'].strip()
        }]

        print("✅ Single speaker transcription completed")
        return transcript

    except Exception as e:
        st.error(f"Failed to transcribe audio: {str(e)}")
        return []

def display_single_speaker_results(transcripts):
    """Display results for single speaker processing"""
    st.header("📋 Analysis Results")

    if not transcripts:
        st.warning("No transcription generated")
        return

    transcript = transcripts[0]

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Recording Type", "Single Speaker")

    with col2:
        st.metric("Total Speakers", "1")

    with col3:
        duration = transcript['end']
        st.metric("Duration", f"{duration:.1f}s")

    # Transcription result
    st.subheader("📝 Complete Transcript")

    duration_fmt = format_time(duration)

    st.markdown(f"""
    <div class="success-box">
        <h4>Speaker_0 (Complete Recording)</h4>
        <p><strong>Duration:</strong> 00:00 - {duration_fmt}</p>
        <p><strong>Transcript:</strong></p>
        <p style="font-style: italic; font-size: 1.1em; line-height: 1.5; color: #1a1a1a;">"{transcript['text']}"</p>
    </div>
    """, unsafe_allow_html=True)

    # Word count and reading time
    word_count = len(transcript['text'].split())
    reading_time = word_count / 200  # Average reading speed

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Word Count", word_count)

    # Create Markdown formatted transcript for download
    markdown_content = f"""# Single Speaker Transcript

## Recording Information
- **Speaker:** Speaker_0
- **Duration:** 00:00 - {duration_fmt}
- **Word Count:** {word_count}
- **Estimated Reading Time:** {reading_time:.1f} minutes

## Transcript

> {transcript['text']}

---
*Generated by VoiceShield Audio Analyzer*
"""

    # Download section - Markdown only
    st.subheader("💾 Download Results")

    st.download_button(
        label="📥 Download Transcript (Markdown)",
        data=markdown_content,
        file_name="single_speaker_transcript.md",
        mime="text/markdown",
        use_container_width=True
    )

def create_overlapping_segments(audio_file, segments_dir, speech_segments, segment_duration=10.0, overlap=2.0):
    """Create overlapping audio segments for better speaker identification"""
    segment_files = []
    y, sr = librosa.load(audio_file, sr=16000)

    segment_idx = 0
    for start_time, end_time in speech_segments:
        # Convert to sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Create segments with overlap
        current_start = start_sample
        while current_start < end_sample:
            current_end = min(current_start + int(segment_duration * sr), end_sample)

            # Only create segment if it's long enough
            if (current_end - current_start) >= sr:  # At least 1 second
                segment_audio = y[current_start:current_end]
                segment_file = segments_dir / f"segment_{segment_idx:04d}.wav"
                sf.write(segment_file, segment_audio, sr)

                segment_files.append({
                    'file': str(segment_file),
                    'start': current_start / sr,
                    'end': current_end / sr,
                    'duration': (current_end - current_start) / sr
                })
                segment_idx += 1

            # Move to next segment with overlap
            current_start += int((segment_duration - overlap) * sr)

    return segment_files

def extract_embeddings(segment_files, diarizer):
    """Extract speaker embeddings from audio segments"""
    embeddings = []
    filenames = []

    progress_container = st.container()
    with progress_container:
        embed_progress = st.progress(0)
        embed_status = st.empty()

    for i, segment_info in enumerate(segment_files):
        embed_status.text(f"Processing segment {i+1}/{len(segment_files)}")

        try:
            embedding = diarizer.extract_speaker_embedding(segment_info['file'])
            if embedding is not None:
                embeddings.append(embedding)
                filenames.append(segment_info)
        except Exception as e:
            print(f"Failed to extract embedding for {segment_info['file']}: {e}")
            continue

        embed_progress.progress((i + 1) / len(segment_files))

    # Clear progress indicators
    embed_progress.empty()
    embed_status.empty()

    return embeddings, filenames

def cluster_speakers(embeddings, filenames, min_speakers, max_speakers):
    """Cluster speaker embeddings to identify different speakers"""
    if len(embeddings) < min_speakers:
        st.warning(f"Only found {len(embeddings)} segments, using single speaker")
        return [(info, 0) for info in filenames]

    embeddings_array = np.array(embeddings)

    best_score = -1
    best_labels = None
    best_n_clusters = min_speakers

    # Try different numbers of clusters
    for n_clusters in range(min_speakers, min(max_speakers + 1, len(embeddings) + 1)):
        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward',
                metric='euclidean'
            )
            labels = clustering.fit_predict(embeddings_array)

            # Calculate silhouette score
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings_array, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_n_clusters = n_clusters
        except Exception as e:
            print(f"Clustering failed for {n_clusters} clusters: {e}")
            continue

    if best_labels is None:
        # Fallback to single speaker
        best_labels = [0] * len(filenames)
        best_n_clusters = 1

    st.info(f"🎯 Identified {best_n_clusters} speakers")

    return list(zip(filenames, best_labels))

def transcribe_segments(audio_file, speaker_labels, temp_path, whisper_model_name, cluster_to_name_mapping=None):
    """Transcribe audio segments with speaker labels"""
    # Determine device for Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print(f"Loading Whisper model on {device}...")
        model = whisper.load_model(whisper_model_name, device=device)
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        return []

    # Group segments by speaker
    speaker_segments = {}
    for segment_info, speaker_id in speaker_labels:
        if speaker_id not in speaker_segments:
            speaker_segments[speaker_id] = []
        speaker_segments[speaker_id].append(segment_info)

    transcripts = []

    # Process each speaker's segments
    progress_container = st.container()
    with progress_container:
        trans_progress = st.progress(0)
        trans_status = st.empty()

    total_segments = len(speaker_labels)
    processed = 0

    for speaker_id, segments in speaker_segments.items():
        trans_status.text(f"Transcribing Speaker_{speaker_id}...")

        for segment_info in segments:
            try:
                # Transcribe individual segment
                result = model.transcribe(segment_info['file'], task="translate")

                if result['text'].strip():
                    transcripts.append({
                        'speaker': f'Speaker_{speaker_id}',
                        'start': segment_info['start'],
                        'end': segment_info['end'],
                        'text': result['text'].strip()
                    })

                processed += 1
                trans_progress.progress(processed / total_segments)

            except Exception as e:
                print(f"Failed to transcribe segment {segment_info['file']}: {e}")
                processed += 1
                trans_progress.progress(processed / total_segments)
                continue

    # Clear progress indicators
    trans_progress.empty()
    trans_status.empty()

    # Sort transcripts by start time
    transcripts.sort(key=lambda x: x['start'])

    return transcripts

def format_time(seconds):
    """Format seconds to MM:SS format"""
    return str(timedelta(seconds=int(seconds)))[2:]  # Remove hours part if 00:

def display_results(transcripts, speaker_labels):
    """Display the final results for multi-speaker mode"""
    st.header("📋 Analysis Results")

    if not transcripts:
        st.warning("No transcription generated")
        return

    # Summary statistics
    speaker_ids = set(label[1] for label in speaker_labels)
    num_speakers = len(speaker_ids)
    total_duration = max(t['end'] for t in transcripts) if transcripts else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Speakers", num_speakers)

    with col2:
        st.metric("Speech Segments", len(transcripts))

    with col3:
        st.metric("Total Duration", f"{total_duration:.1f}s")

    # Speaker distribution
    if num_speakers > 1:
        st.subheader("👥 Speaker Distribution")

        speaker_stats = {}
        for transcript in transcripts:
            speaker = transcript['speaker']
            duration = transcript['end'] - transcript['start']

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {'segments': 0, 'duration': 0}

            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['duration'] += duration

        # Create distribution chart
        speakers = list(speaker_stats.keys())
        durations = [speaker_stats[s]['duration'] for s in speakers]

        chart_data = pd.DataFrame({
            'Speaker': speakers,
            'Duration (seconds)': durations
        })

        st.bar_chart(chart_data.set_index('Speaker'))

        # Detailed speaker stats
        for speaker, stats in speaker_stats.items():
            percentage = (stats['duration'] / total_duration) * 100
            st.write(f"**{speaker}:** {stats['segments']} segments, "
                    f"{stats['duration']:.1f}s ({percentage:.1f}%)")

    # Full transcript
    st.subheader("📝 Complete Transcript")

    # Merge consecutive segments from same speaker
    merged_transcripts = []
    current_speaker = None
    current_text = ""
    current_start = None
    current_end = None

    for transcript in transcripts:
        if transcript['speaker'] != current_speaker:
            # Save previous merged segment
            if current_speaker is not None:
                merged_transcripts.append({
                    'speaker': current_speaker,
                    'start': current_start,
                    'end': current_end,
                    'text': current_text.strip()
                })

            # Start new segment
            current_speaker = transcript['speaker']
            current_text = transcript['text']
            current_start = transcript['start']
            current_end = transcript['end']
        else:
            # Merge with current segment
            current_text += " " + transcript['text']
            current_end = transcript['end']

    # Don't forget the last segment
    if current_speaker is not None:
        merged_transcripts.append({
            'speaker': current_speaker,
            'start': current_start,
            'end': current_end,
            'text': current_text.strip()
        })

    # Display merged transcript with improved text contrast
    for transcript in merged_transcripts:
        start_time = format_time(transcript['start'])
        end_time = format_time(transcript['end'])

        # Color coding for different speakers
        speaker_colors = {
            'Speaker_0': '#e8f4fd',
            'Speaker_1': '#fff2e8',
            'Speaker_2': '#f0f8e8',
            'Speaker_3': '#fdf0f8',
            'Speaker_4': '#f8f0fd',
            'Speaker_5': '#e8f8f5',
            'Speaker_6': '#fdf8e8',
            'Speaker_7': '#f0e8fd'
        }

        bg_color = speaker_colors.get(transcript['speaker'], '#f8f9fa')

        st.markdown(f"""
        <div style="background-color: {bg_color}; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #007bff;">
            <h4 style="margin: 0 0 0.5rem 0; color: #333;">{transcript['speaker']}</h4>
            <p style="margin: 0 0 0.5rem 0; font-size: 0.9em; color: #666;"><strong>Time:</strong> {start_time} - {end_time}</p>
            <p style="margin: 0; font-style: italic; line-height: 1.5; color: #1a1a1a;">"{transcript['text']}"</p>
        </div>
        """, unsafe_allow_html=True)

    # Create Markdown formatted transcript for multi-speaker download
    markdown_lines = ["# Multi-Speaker Transcript\n"]

    # Add metadata
    markdown_lines.append("## Recording Information")
    markdown_lines.append(f"- **Total Speakers:** {num_speakers}")
    markdown_lines.append(f"- **Speech Segments:** {len(transcripts)}")
    markdown_lines.append(f"- **Total Duration:** {total_duration:.1f}s\n")

    # Add speaker distribution if multiple speakers
    if num_speakers > 1:
        markdown_lines.append("## Speaker Distribution")
        for speaker, stats in speaker_stats.items():
            percentage = (stats['duration'] / total_duration) * 100
            markdown_lines.append(f"- **{speaker}:** {stats['segments']} segments, {stats['duration']:.1f}s ({percentage:.1f}%)")
        markdown_lines.append("")

    # Add transcript
    markdown_lines.append("## Transcript\n")

    for transcript in merged_transcripts:
        start_time = format_time(transcript['start'])
        end_time = format_time(transcript['end'])
        markdown_lines.append(f"### {transcript['speaker']}")
        markdown_lines.append(f"**Time:** {start_time} - {end_time}")
        markdown_lines.append(f"> {transcript['text']}\n")

    markdown_lines.append("---")
    markdown_lines.append("*Generated by VoiceShield Audio Analyzer*")

    markdown_content = "\n".join(markdown_lines)

    # Download section - Markdown only
    st.subheader("💾 Download Results")

    st.download_button(
        label="📥 Download Transcript (Markdown)",
        data=markdown_content,
        file_name="multi_speaker_transcript.md",
        mime="text/markdown",
        use_container_width=True
    )

# Custom error handling and cleanup
def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

# Main execution
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.info("Processing interrupted by user")
        cleanup_gpu_memory()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        cleanup_gpu_memory()
        raise
    finally:
        # Final cleanup
        cleanup_gpu_memory()
