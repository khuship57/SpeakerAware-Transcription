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
import subprocess
import sys
import shutil

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
    page_title="Audio Processing Pipeline",
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
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AudioProcessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
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
        """Load audio from a file with multiple fallback methods."""
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

    def normalize(self, audio: np.ndarray, method: str = 'peak') -> np.ndarray:
        """Normalize audio signal."""
        if method == 'peak':
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = 0.95 * audio / peak
        elif method == 'rms':
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio * (0.1 / rms)
        elif method == 'lufs':
            loudness = np.mean(np.abs(audio))
            if loudness > 0:
                audio = audio * (0.1 / loudness)
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

        audio = self.normalize(audio, method=kwargs.get('normalize_method', 'peak'))
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
    st.title("🎵 Audio Processing Pipeline")
    st.markdown("### Complete pipeline for audio preprocessing, speaker diarization, and transcription")

    # Check FFmpeg installation at startup
    if not setup_ffmpeg():
        st.error("⚠️ FFmpeg setup incomplete. Some audio formats may not be supported.")
        st.info("You can still try uploading WAV files directly, or install FFmpeg manually.")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Speaker Diarization options
    st.sidebar.subheader("Speaker Diarization")
    min_speakers = st.sidebar.number_input("Minimum Speakers", min_value=1, max_value=10, value=2)
    max_speakers = st.sidebar.number_input("Maximum Speakers", min_value=1, max_value=10, value=8)

    st.sidebar.subheader("Whisper Transcription")
    whisper_model_name = st.sidebar.selectbox(
        "Choose Whisper model",
        options=["tiny", "base", "small", "medium", "large"],
        index=2  # default to "small"
    )

    # File upload
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
    )

    if uploaded_file is not None:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save uploaded file
            input_file = temp_path / uploaded_file.name
            with open(input_file, "wb") as f:
                f.write(uploaded_file.read())

            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # Display file info
            file_stats = os.stat(input_file)
            st.info(f"📊 File size: {file_stats.st_size / (1024*1024):.2f} MB")

            # Audio preview
            st.audio(str(input_file))

            if st.button("🚀 Start Processing", type="primary"):
                process_audio_pipeline(
                    input_file=input_file,
                    temp_path=temp_path,
                    normalize_method="peak", 
                    apply_highpass=True,      
                    apply_denoising=True,     
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    whisper_model_name=whisper_model_name
                )

def process_audio_pipeline(input_file, temp_path, normalize_method, apply_highpass,
                          apply_denoising, min_speakers, max_speakers, whisper_model_name):
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

        st.info(f"✅ {message}")
        progress_bar.progress(10)

        # Step 2: Preprocess audio
        status_text.text("🔄 Preprocessing audio...")
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

        # Step 3: Voice Activity Detection (CPU ONLY)
        if not SPEECHBRAIN_AVAILABLE:
            st.error("❌ SpeechBrain not available. Cannot proceed with speaker diarization.")
            return

        status_text.text("🔄 Detecting speech segments...")
        diarizer = SpeakerDiarization()

        if not diarizer.load_vad_model():
            st.error("❌ Failed to load VAD model")
            return

        speech_segments = diarizer.get_speech_segments(str(preprocessed_file))
        st.info(f"🎯 Found {len(speech_segments)} speech segments")
        progress_bar.progress(40)

        # Step 4: Create overlapping segments
        status_text.text("🔄 Creating overlapping segments...")
        segments_dir = temp_path / "segments"
        segments_dir.mkdir(exist_ok=True)

        segment_files = create_overlapping_segments(
            str(preprocessed_file), segments_dir, speech_segments
        )
        progress_bar.progress(55)

        # Step 5: Extract speaker embeddings (GPU if available)
        status_text.text(f"🔄 Extracting speaker embeddings...")
        
        if not diarizer.load_speaker_model():
            st.error("❌ Failed to load speaker model")
            return

        embeddings, filenames = extract_embeddings(segment_files, diarizer)
        if len(embeddings) == 0:
            st.error("❌ No valid embeddings extracted")
            return

        progress_bar.progress(70)

        # Step 6: Speaker clustering (CPU)
        status_text.text("🔄 Clustering speakers...")
        speaker_labels = cluster_speakers(
            embeddings, filenames, min_speakers, max_speakers
        )
        progress_bar.progress(85)

        # Step 7: Transcription (GPU if available for Whisper)
        whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        status_text.text(f"🔄 Transcribing audio...")
        transcripts = transcribe_segments(
            str(preprocessed_file), speaker_labels, temp_path, whisper_model_name, whisper_device
        )
        progress_bar.progress(100)

        # Display results
        status_text.text("✅ Processing complete!")
        display_results(transcripts, speaker_labels)

    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.exception(e)
        
        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def create_overlapping_segments(audio_file, output_dir, speech_segments,
                               window_duration=2.0, overlap=0.5):
    """Create overlapping segments from speech regions"""
    y, sr = librosa.load(audio_file, sr=16000)
    window_samples = int(window_duration * sr)
    hop_samples = int((window_duration - overlap) * sr)

    segment_files = []
    segment_id = 0

    for start_time, end_time in speech_segments:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = y[start_sample:end_sample]

        # Create overlapping windows within this speech segment
        for i in range(0, len(segment_audio) - window_samples + 1, hop_samples):
            chunk = segment_audio[i:i + window_samples]
            if len(chunk) == window_samples:
                chunk_start_time = start_time + (i / sr)
                chunk_end_time = start_time + ((i + window_samples) / sr)

                filename = f"segment{segment_id:03d}_{chunk_start_time:.2f}_{chunk_end_time:.2f}.wav"
                filepath = output_dir / filename
                sf.write(filepath, chunk, sr)
                segment_files.append(str(filepath))
                segment_id += 1

    return segment_files

def extract_embeddings(segment_files, diarizer):
    """Extract speaker embeddings from segment files"""
    embeddings = []
    filenames = []

    print(f"Extracting embeddings from {len(segment_files)} segments...")
    
    for i, file_path in enumerate(segment_files):
        if i % 10 == 0:  # Progress update
            print(f"Processing segment {i+1}/{len(segment_files)}")
            
        embedding = diarizer.extract_speaker_embedding(file_path)
        if embedding is not None:
            embeddings.append(embedding)
            filenames.append(Path(file_path).name)

    print(f"Successfully extracted {len(embeddings)} embeddings")
    return np.array(embeddings) if embeddings else np.array([]), filenames

def cluster_speakers(embeddings, filenames, min_speakers, max_speakers):
    """Cluster speaker embeddings to identify different speakers"""
    if len(embeddings) == 0:
        return {}

    print(f"Clustering {len(embeddings)} embeddings...")

    # Find optimal number of clusters
    best_score = -1
    best_labels = None
    best_n_clusters = min_speakers

    for n_clusters in range(min_speakers, min(max_speakers + 1, len(embeddings))):
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)

        if len(set(labels)) > 1:
            score = silhouette_score(embeddings, labels, metric='cosine')
            if score > best_score:
                best_score = score
                best_labels = labels
                best_n_clusters = n_clusters

    if best_labels is None:
        best_labels = [0] * len(filenames)
        best_n_clusters = 1

    # Create mapping
    speaker_labels = {}
    for filename, label in zip(filenames, best_labels):
        speaker_labels[filename] = f"Speaker_{label}"

    print(f"Identified {best_n_clusters} speakers with silhouette score: {best_score:.3f}")
    st.info(f"🎯 Identified {best_n_clusters} speakers")

    return speaker_labels

def transcribe_segments(audio_file, speaker_labels, temp_path, whisper_model_name, device="cpu"):
    """Transcribe audio segments with speaker labels - optimized for GPU"""
    # Load Whisper model with device specification
    try:
        print(f"Loading Whisper model on {device}...")
        model = whisper.load_model(whisper_model_name, device=device)
        print("✅ Whisper model loaded successfully")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        return []

    # Parse segments and merge by speaker
    segments = []
    for filename, speaker in speaker_labels.items():
        match = re.search(r'_(\d+\.\d+)_(\d+\.\d+)\.wav', filename)
        if match:
            start_time = float(match.group(1))
            end_time = float(match.group(2))
            segments.append({
                'filename': filename,
                'start': start_time,
                'end': end_time,
                'speaker': speaker
            })

    # Sort by time and merge consecutive segments of same speaker
    segments.sort(key=lambda x: x['start'])
    merged_segments = merge_speaker_segments(segments)

    print(f"Transcribing {len(merged_segments)} merged segments...")

    # Extract merged segments and transcribe
    transcripts = []
    for i, seg in enumerate(merged_segments):
        print(f"Transcribing segment {i+1}/{len(merged_segments)}")
        
        # Extract audio segment
        y, sr = librosa.load(audio_file, sr=16000)
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        segment_audio = y[start_sample:end_sample]

        # Save temporary segment
        temp_segment = temp_path / f"temp_segment_{i}.wav"
        sf.write(temp_segment, segment_audio, sr)

        # Transcribe
        try:
            result = model.transcribe(str(temp_segment), task="transcribe")
            text = result['text'].strip()

            if text: 
                transcripts.append({
                    'speaker': seg['speaker'],
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': text
                })

        except Exception as e:
            st.warning(f"Failed to transcribe segment {i}: {str(e)}")

    print(f"✅ Transcription completed: {len(transcripts)} segments transcribed")
    return transcripts

def merge_speaker_segments(segments, max_gap=1.0):
    """Merge consecutive segments from the same speaker"""
    if not segments:
        return []

    merged = []
    current = segments[0].copy()

    for seg in segments[1:]:
        if (seg['speaker'] == current['speaker'] and
            seg['start'] - current['end'] <= max_gap):
            # Merge segments
            current['end'] = seg['end']
        else:
            merged.append(current)
            current = seg.copy()

    merged.append(current)
    return merged

def format_time(seconds):
    td = timedelta(seconds=round(seconds))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{secs:02}"
    else:
        return f"{minutes:02}:{secs:02}"

def display_results(transcripts, speaker_labels):
    """Display processing results"""
    st.header("📋 Results")

    # Summary statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Segments", len(speaker_labels))

    with col2:
        unique_speakers = len(set(speaker_labels.values()))
        st.metric("Speakers Identified", unique_speakers)

    with col3:
        total_duration = max([t['end'] for t in transcripts]) if transcripts else 0
        st.metric("Total Duration", f"{total_duration:.1f}s")

    # Speaker distribution
    if speaker_labels:
        st.subheader("🎭 Speaker Distribution")
        speaker_counts = {}
        for speaker in speaker_labels.values():
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        df_speakers = pd.DataFrame(list(speaker_counts.items()),
                                  columns=['Speaker', 'Segments'])
        st.bar_chart(df_speakers.set_index('Speaker'))

    # Transcription results
    if transcripts:
        st.subheader("📝 Transcript with Speaker Labels")

        # Create downloadable transcript
        transcript_text = ""
        for t in transcripts:
            start_fmt = format_time(t['start'])
            end_fmt = format_time(t['end'])
            line = f"[{t['speaker']}] [ {start_fmt} - {end_fmt} ] {t['text']}\n"
            transcript_text += line

        # Display transcript
        for t in transcripts:
            start_fmt = format_time(t['start'])
            end_fmt = format_time(t['end'])
            st.markdown(f"""
            **{t['speaker']}** `[ {start_fmt} - {end_fmt} ]`  
            {t['text']}""")

        # Download button
        st.download_button(
            label="📥 Download Transcript",
            data=transcript_text,
            file_name="transcript.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    main()
