"""
Audio preprocessing pipeline for Voice Lab GPT
Handles audio loading, normalization, and preprocessing
"""

import numpy as np
import librosa
import pyloudnorm as pyln
from pydub import AudioSegment
from scipy import signal
from typing import Tuple, Optional, Dict, Any
import warnings

class AudioProcessor:
    """Handles all audio preprocessing operations"""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 target_lufs: float = -23.0,
                 silence_threshold_db: float = -40.0):
        """
        Initialize audio processor
        
        Args:
            target_sr: Target sampling rate
            target_lufs: Target loudness normalization level
            silence_threshold_db: Threshold for silence removal
        """
        self.target_sr = target_sr
        self.target_lufs = target_lufs
        self.silence_threshold_db = silence_threshold_db
        self.meter = pyln.Meter(target_sr)
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Load audio file and extract metadata
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate, metadata)
        """
        try:
            # Load with librosa for better format support
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            
            # Get file info using pydub for metadata
            audio_segment = AudioSegment.from_file(file_path)
            
            metadata = {
                'original_sr': sr,
                'channels': audio_segment.channels,
                'duration': len(audio_segment) / 1000.0,  # seconds
                'format': file_path.split('.')[-1].lower(),
                'frame_rate': audio_segment.frame_rate,
                'sample_width': audio_segment.sample_width
            }
            
            return audio, sr, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")
    
    def preprocess_audio(self, 
                        audio: np.ndarray, 
                        sr: int,
                        convert_mono: bool = True,
                        normalize_loudness: bool = True,
                        remove_silence: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete audio preprocessing pipeline
        
        Args:
            audio: Audio signal
            sr: Sample rate
            convert_mono: Convert to mono if True
            normalize_loudness: Apply loudness normalization if True
            remove_silence: Remove leading/trailing silence if True
            
        Returns:
            Tuple of (processed_audio, processing_info)
        """
        processing_info = {
            'original_shape': audio.shape,
            'original_sr': sr,
            'steps_applied': []
        }
        
        # Convert to mono if needed
        if convert_mono and audio.ndim > 1:
            audio = self._convert_to_mono(audio)
            processing_info['steps_applied'].append('mono_conversion')
            
        # Resample to target sample rate
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
            processing_info['steps_applied'].append(f'resampling_to_{self.target_sr}')
        
        # Convert to 16-bit equivalent range
        audio = self._ensure_16bit_range(audio)
        processing_info['steps_applied'].append('16bit_conversion')
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        processing_info['steps_applied'].append('dc_removal')
        
        # Remove leading/trailing silence
        if remove_silence:
            audio, trim_info = self._remove_silence(audio, sr)
            processing_info['trim_info'] = trim_info
            processing_info['steps_applied'].append('silence_removal')
        
        # Loudness normalization
        if normalize_loudness:
            audio, norm_info = self._normalize_loudness(audio)
            processing_info['loudness_info'] = norm_info
            processing_info['steps_applied'].append('loudness_normalization')
        
        # Final quality checks
        quality_metrics = self._assess_quality(audio, sr)
        processing_info['quality_metrics'] = quality_metrics
        
        processing_info['final_shape'] = audio.shape
        processing_info['final_duration'] = len(audio) / sr
        
        return audio, processing_info
    
    def _convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono"""
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            return np.mean(audio, axis=0)
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")
    
    def _ensure_16bit_range(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is in appropriate range for 16-bit processing"""
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
        elif max_val > 0:
            # Scale to use full range but avoid clipping
            audio = audio * 0.95
        return audio
    
    def _remove_silence(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove leading and trailing silence"""
        # Use librosa's trim function with energy-based detection
        audio_trimmed, trim_indices = librosa.effects.trim(
            audio, 
            top_db=-self.silence_threshold_db,
            frame_length=512,
            hop_length=128
        )
        
        trim_info = {
            'samples_removed_start': trim_indices[0],
            'samples_removed_end': len(audio) - trim_indices[1],
            'seconds_removed_start': trim_indices[0] / sr,
            'seconds_removed_end': (len(audio) - trim_indices[1]) / sr,
            'original_length': len(audio),
            'trimmed_length': len(audio_trimmed)
        }
        
        return audio_trimmed, trim_info
    
    def _normalize_loudness(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply loudness normalization using ITU-R BS.1770 standard"""
        try:
            # Measure current loudness
            current_loudness = self.meter.integrated_loudness(audio)
            
            # Apply loudness normalization
            if np.isfinite(current_loudness):
                audio_normalized = pyln.normalize.loudness(
                    audio, current_loudness, self.target_lufs
                )
                
                # Ensure no clipping
                max_val = np.max(np.abs(audio_normalized))
                if max_val > 0.95:
                    audio_normalized = audio_normalized * (0.95 / max_val)
                    
                norm_info = {
                    'original_loudness_lufs': current_loudness,
                    'target_loudness_lufs': self.target_lufs,
                    'normalization_applied': True,
                    'peak_after_norm': max_val
                }
                
                return audio_normalized, norm_info
            else:
                warnings.warn("Could not measure loudness, skipping normalization")
                return audio, {
                    'normalization_applied': False,
                    'reason': 'unmeasurable_loudness'
                }
                
        except Exception as e:
            warnings.warn(f"Loudness normalization failed: {str(e)}")
            return audio, {
                'normalization_applied': False,
                'reason': f'error: {str(e)}'
            }
    
    def _assess_quality(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Assess audio quality metrics"""
        # Signal-to-noise ratio estimation
        signal_power = np.mean(audio ** 2)
        
        # Estimate noise from quiet segments
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        frames = librosa.util.frame(audio, frame_length=frame_length, 
                                  hop_length=hop_length, axis=0)
        frame_energy = np.mean(frames ** 2, axis=0)
        
        # Use bottom 10% of frames as noise estimate
        noise_threshold = np.percentile(frame_energy, 10)
        estimated_snr = 10 * np.log10(signal_power / max(noise_threshold, 1e-10))
        
        # Check for clipping
        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
        clipping_percentage = (clipped_samples / len(audio)) * 100
        
        # Duration check
        duration = len(audio) / sr
        
        quality_metrics = {
            'estimated_snr_db': estimated_snr,
            'clipped_samples': clipped_samples,
            'clipping_percentage': clipping_percentage,
            'duration_seconds': duration,
            'rms_level': np.sqrt(np.mean(audio ** 2)),
            'peak_level': np.max(np.abs(audio)),
            'is_good_quality': (
                estimated_snr > 20 and 
                clipping_percentage < 0.1 and 
                duration >= 1.0
            )
        }
        
        return quality_metrics
    
    def process_file(self, file_path: str, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete pipeline: load and process audio file
        
        Args:
            file_path: Path to audio file
            **kwargs: Arguments for preprocess_audio
            
        Returns:
            Tuple of (processed_audio, complete_info)
        """
        # Load audio
        audio, sr, metadata = self.load_audio(file_path)
        
        # Process audio
        processed_audio, processing_info = self.preprocess_audio(audio, sr, **kwargs)
        
        # Combine information
        complete_info = {
            'file_metadata': metadata,
            'processing_info': processing_info
        }
        
        return processed_audio, complete_info