"""
Acoustic feature extraction for Voice Lab GPT
Implements all acoustic measures as specified in the system prompt
"""

import numpy as np
import librosa
import parselmouth
from parselmouth import praat
import scipy.signal
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

class AcousticAnalyzer:
    """Extracts comprehensive acoustic features from voice signals"""
    
    def __init__(self, sr: int = 16000):
        """
        Initialize acoustic analyzer
        
        Args:
            sr: Sample rate of audio
        """
        self.sr = sr
        
    def extract_all_features(self, 
                           audio: np.ndarray, 
                           f0_min: float = 60.0,
                           f0_max: float = 400.0,
                           voice_type: Optional[str] = None) -> Dict[str, any]:
        """
        Extract all acoustic features
        
        Args:
            audio: Audio signal
            f0_min: Minimum F0 for analysis
            f0_max: Maximum F0 for analysis
            voice_type: 'male', 'female', or None for auto-detection
            
        Returns:
            Dictionary of all acoustic features
        """
        # Adjust F0 range based on voice type
        if voice_type == 'male':
            f0_min, f0_max = max(f0_min, 60), min(f0_max, 250)
        elif voice_type == 'female':
            f0_min, f0_max = max(f0_min, 100), min(f0_max, 400)
            
        features = {}
        
        # Create Praat sound object
        try:
            sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
        except Exception as e:
            raise ValueError(f"Failed to create Praat sound object: {str(e)}")
        
        # Fundamental frequency analysis
        f0_features = self._extract_f0_features(sound, f0_min, f0_max)
        features.update(f0_features)
        
        # Jitter and shimmer
        periodicity_features = self._extract_periodicity_features(sound, f0_min, f0_max)
        features.update(periodicity_features)
        
        # Harmonics-to-noise ratio
        hnr_features = self._extract_hnr_features(sound, f0_min, f0_max)
        features.update(hnr_features)
        
        # Cepstral peak prominence
        cpp_features = self._extract_cpp_features(audio)
        features.update(cpp_features)
        
        # Spectral tilt
        spectral_features = self._extract_spectral_features(audio)
        features.update(spectral_features)
        
        # Intensity measures
        intensity_features = self._extract_intensity_features(sound)
        features.update(intensity_features)
        
        # Formant analysis (if applicable)
        formant_features = self._extract_formant_features(sound, voice_type)
        features.update(formant_features)
        
        # Voice quality indices
        quality_features = self._extract_voice_quality_features(audio, features)
        features.update(quality_features)
        
        return features
    
    def _extract_f0_features(self, sound, f0_min: float, f0_max: float) -> Dict[str, float]:
        """Extract fundamental frequency features"""
        try:
            # Praat pitch analysis
            pitch = sound.to_pitch_ac(
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
                time_step=0.01,
                very_accurate=True
            )
            
            pitch_values = pitch.selected_array['frequency']
            voiced_frames = pitch_values[pitch_values != 0]
            
            if len(voiced_frames) == 0:
                return {
                    'f0_mean': 0,
                    'f0_median': 0,
                    'f0_std': 0,
                    'f0_min': 0,
                    'f0_max': 0,
                    'voiced_fraction': 0,
                    'f0_semitone_std': 0
                }
            
            # Convert to semitones for relative measures
            semitones = 12 * np.log2(voiced_frames / np.mean(voiced_frames))
            
            return {
                'f0_mean': float(np.mean(voiced_frames)),
                'f0_median': float(np.median(voiced_frames)),
                'f0_std': float(np.std(voiced_frames)),
                'f0_min': float(np.min(voiced_frames)),
                'f0_max': float(np.max(voiced_frames)),
                'voiced_fraction': len(voiced_frames) / len(pitch_values),
                'f0_semitone_std': float(np.std(semitones))
            }
            
        except Exception as e:
            warnings.warn(f"F0 extraction failed: {str(e)}")
            return {key: 0 for key in ['f0_mean', 'f0_median', 'f0_std', 
                                     'f0_min', 'f0_max', 'voiced_fraction', 'f0_semitone_std']}
    
    def _extract_periodicity_features(self, sound, f0_min: float, f0_max: float) -> Dict[str, float]:
        """Extract jitter and shimmer measures"""
        try:
            # Create point process for jitter/shimmer analysis
            pitch = sound.to_pitch_ac(pitch_floor=f0_min, pitch_ceiling=f0_max)
            point_process = praat.call([sound, pitch], "To PointProcess (cc)")
            
            # Jitter measures
            jitter_local = praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Shimmer measures  
            shimmer_local = praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = praat.call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = praat.call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq11 = praat.call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            return {
                'jitter_local': float(jitter_local if np.isfinite(jitter_local) else 0),
                'jitter_rap': float(jitter_rap if np.isfinite(jitter_rap) else 0),
                'jitter_ppq5': float(jitter_ppq5 if np.isfinite(jitter_ppq5) else 0),
                'shimmer_local': float(shimmer_local if np.isfinite(shimmer_local) else 0),
                'shimmer_apq3': float(shimmer_apq3 if np.isfinite(shimmer_apq3) else 0),
                'shimmer_apq5': float(shimmer_apq5 if np.isfinite(shimmer_apq5) else 0),
                'shimmer_apq11': float(shimmer_apq11 if np.isfinite(shimmer_apq11) else 0)
            }
            
        except Exception as e:
            warnings.warn(f"Jitter/shimmer extraction failed: {str(e)}")
            return {key: 0 for key in ['jitter_local', 'jitter_rap', 'jitter_ppq5',
                                     'shimmer_local', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11']}
    
    def _extract_hnr_features(self, sound, f0_min: float, f0_max: float) -> Dict[str, float]:
        """Extract harmonics-to-noise ratio and noise-to-harmonics ratio"""
        try:
            # HNR using Praat's harmonicity
            harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=f0_min, silence_threshold=0.1, periods_per_window=1.0)
            hnr_values = harmonicity.values[harmonicity.values != -200]  # Remove undefined values
            
            if len(hnr_values) == 0:
                return {'hnr_mean': 0, 'hnr_std': 0, 'nhr_mean': np.inf}
            
            hnr_mean = float(np.mean(hnr_values))
            hnr_std = float(np.std(hnr_values))
            
            # NHR is inverse of HNR (in linear scale)
            hnr_linear = 10 ** (hnr_values / 10)
            nhr_values = 1 / hnr_linear
            nhr_mean = float(np.mean(nhr_values))
            
            return {
                'hnr_mean': hnr_mean,
                'hnr_std': hnr_std,
                'nhr_mean': nhr_mean
            }
            
        except Exception as e:
            warnings.warn(f"HNR extraction failed: {str(e)}")
            return {'hnr_mean': 0, 'hnr_std': 0, 'nhr_mean': np.inf}
    
    def _extract_cpp_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract Cepstral Peak Prominence"""
        try:
            # Frame-based analysis
            frame_length = int(0.04 * self.sr)  # 40ms frames
            hop_length = int(0.01 * self.sr)    # 10ms hop
            
            frames = librosa.util.frame(audio, frame_length=frame_length, 
                                      hop_length=hop_length, axis=0).T
            
            cpp_values = []
            cpps_values = []
            
            for frame in frames:
                if np.sum(frame**2) < 1e-10:  # Skip silent frames
                    continue
                    
                # Compute power spectrum
                fft = np.fft.fft(frame * np.hanning(len(frame)))
                power_spectrum = np.abs(fft[:len(fft)//2])**2
                
                # Compute cepstrum
                log_spectrum = np.log(power_spectrum + 1e-10)
                cepstrum = np.fft.fft(log_spectrum)
                quefrency = np.arange(len(cepstrum)) / self.sr
                
                # Find peak in expected F0 range (corresponding to quefrency)
                min_quefrency = 1/400  # 400 Hz max
                max_quefrency = 1/60   # 60 Hz min
                
                valid_indices = (quefrency >= min_quefrency) & (quefrency <= max_quefrency)
                if np.sum(valid_indices) > 0:
                    cepstral_peak = np.max(np.real(cepstrum[valid_indices]))
                    cepstral_mean = np.mean(np.real(cepstrum[valid_indices]))
                    
                    cpp = cepstral_peak - cepstral_mean
                    cpp_values.append(cpp)
                    
                    # CPPS: smoothed version
                    smoothed_cepstrum = scipy.signal.savgol_filter(
                        np.real(cepstrum[valid_indices]), 
                        min(11, len(cepstrum[valid_indices])//2*2+1), 2
                    )
                    cpps_peak = np.max(smoothed_cepstrum)
                    cpps = cpps_peak - cepstral_mean
                    cpps_values.append(cpps)
            
            if len(cpp_values) == 0:
                return {'cpp_mean': 0, 'cpp_std': 0, 'cpps_mean': 0, 'cpps_std': 0}
            
            return {
                'cpp_mean': float(np.mean(cpp_values)),
                'cpp_std': float(np.std(cpp_values)),
                'cpps_mean': float(np.mean(cpps_values)),
                'cpps_std': float(np.std(cpps_values))
            }
            
        except Exception as e:
            warnings.warn(f"CPP extraction failed: {str(e)}")
            return {'cpp_mean': 0, 'cpp_std': 0, 'cpps_mean': 0, 'cpps_std': 0}
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral tilt and related measures"""
        try:
            # Compute power spectral density
            frequencies, psd = scipy.signal.welch(audio, self.sr, nperseg=1024)
            
            # Convert to dB
            psd_db = 10 * np.log10(psd + 1e-10)
            
            # Spectral tilt: linear regression slope on log spectrum
            # Focus on 0-5000 Hz range
            max_freq = min(5000, self.sr // 2)
            valid_indices = frequencies <= max_freq
            
            log_freq = np.log10(frequencies[valid_indices] + 1)
            spectral_slope, _, r_value, _, _ = stats.linregress(log_freq, psd_db[valid_indices])
            
            # Spectral centroid
            spectral_centroid = np.sum(frequencies * psd) / np.sum(psd)
            
            # Spectral rolloff (95% energy point)
            cumsum_psd = np.cumsum(psd)
            rolloff_95 = frequencies[np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]]
            
            # High frequency ratio (energy above 1000 Hz / total energy)
            high_freq_idx = frequencies >= 1000
            if np.sum(high_freq_idx) > 0:
                high_freq_ratio = np.sum(psd[high_freq_idx]) / np.sum(psd)
            else:
                high_freq_ratio = 0
            
            return {
                'spectral_tilt': float(spectral_slope),
                'spectral_tilt_r2': float(r_value**2),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff_95': float(rolloff_95),
                'high_freq_ratio': float(high_freq_ratio)
            }
            
        except Exception as e:
            warnings.warn(f"Spectral feature extraction failed: {str(e)}")
            return {key: 0 for key in ['spectral_tilt', 'spectral_tilt_r2', 
                                     'spectral_centroid', 'spectral_rolloff_95', 'high_freq_ratio']}
    
    def _extract_intensity_features(self, sound) -> Dict[str, float]:
        """Extract intensity-related features"""
        try:
            intensity = sound.to_intensity(time_step=0.01, minimum_pitch=50)
            intensity_values = intensity.values[intensity.values != -200]  # Remove undefined
            
            if len(intensity_values) == 0:
                return {'intensity_mean': 0, 'intensity_std': 0, 'intensity_range': 0}
            
            return {
                'intensity_mean': float(np.mean(intensity_values)),
                'intensity_std': float(np.std(intensity_values)),
                'intensity_range': float(np.max(intensity_values) - np.min(intensity_values))
            }
            
        except Exception as e:
            warnings.warn(f"Intensity extraction failed: {str(e)}")
            return {'intensity_mean': 0, 'intensity_std': 0, 'intensity_range': 0}
    
    def _extract_formant_features(self, sound, voice_type: Optional[str]) -> Dict[str, float]:
        """Extract formant frequencies (F1, F2, F3)"""
        try:
            # Adjust formant settings based on voice type
            if voice_type == 'male':
                max_formant = 5000
            elif voice_type == 'female':
                max_formant = 5500
            else:
                max_formant = 5200  # Default
            
            formants = sound.to_formant_burg(
                time_step=0.01,
                max_number_of_formants=5,
                maximum_formant=max_formant,
                window_length=0.025,
                pre_emphasis_from=50
            )
            
            f1_values = []
            f2_values = []
            f3_values = []
            
            for i in range(formants.get_number_of_frames()):
                f1 = formants.get_value_at_time(1, formants.get_time_from_frame_number(i+1))
                f2 = formants.get_value_at_time(2, formants.get_time_from_frame_number(i+1))
                f3 = formants.get_value_at_time(3, formants.get_time_from_frame_number(i+1))
                
                if not np.isnan(f1) and f1 > 0:
                    f1_values.append(f1)
                if not np.isnan(f2) and f2 > 0:
                    f2_values.append(f2)
                if not np.isnan(f3) and f3 > 0:
                    f3_values.append(f3)
            
            # Calculate vowel space area if we have F1 and F2
            vowel_space_area = 0
            if len(f1_values) > 2 and len(f2_values) > 2:
                # Simplified vowel space calculation
                f1_range = np.max(f1_values) - np.min(f1_values)
                f2_range = np.max(f2_values) - np.min(f2_values)
                vowel_space_area = f1_range * f2_range
            
            return {
                'f1_mean': float(np.mean(f1_values)) if f1_values else 0,
                'f1_std': float(np.std(f1_values)) if f1_values else 0,
                'f2_mean': float(np.mean(f2_values)) if f2_values else 0,
                'f2_std': float(np.std(f2_values)) if f2_values else 0,
                'f3_mean': float(np.mean(f3_values)) if f3_values else 0,
                'f3_std': float(np.std(f3_values)) if f3_values else 0,
                'vowel_space_area': float(vowel_space_area),
                'formant_frames_detected': len(f1_values)
            }
            
        except Exception as e:
            warnings.warn(f"Formant extraction failed: {str(e)}")
            return {key: 0 for key in ['f1_mean', 'f1_std', 'f2_mean', 'f2_std', 
                                     'f3_mean', 'f3_std', 'vowel_space_area', 'formant_frames_detected']}
    
    def _extract_voice_quality_features(self, audio: np.ndarray, features: Dict) -> Dict[str, float]:
        """Extract additional voice quality indices"""
        try:
            # Pause-to-speech ratio (simplified)
            frame_length = int(0.025 * self.sr)
            hop_length = int(0.01 * self.sr)
            
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length, axis=0).T
            frame_energy = np.mean(frames**2, axis=1)
            
            # Voice activity detection using energy threshold
            energy_threshold = np.percentile(frame_energy, 30)
            voiced_frames = frame_energy > energy_threshold
            pause_ratio = 1 - (np.sum(voiced_frames) / len(voiced_frames))
            
            # Dynamic range
            rms_values = np.sqrt(frame_energy)
            if len(rms_values) > 0:
                dynamic_range = 20 * np.log10(np.max(rms_values) / (np.min(rms_values) + 1e-10))
            else:
                dynamic_range = 0
                
            # Roughness index (based on F0 and amplitude modulation)
            roughness_index = features.get('jitter_local', 0) + features.get('shimmer_local', 0)
            
            return {
                'pause_ratio': float(pause_ratio),
                'dynamic_range_db': float(dynamic_range),
                'roughness_index': float(roughness_index),
                'voice_breaks': self._count_voice_breaks(voiced_frames)
            }
            
        except Exception as e:
            warnings.warn(f"Voice quality feature extraction failed: {str(e)}")
            return {'pause_ratio': 0, 'dynamic_range_db': 0, 'roughness_index': 0, 'voice_breaks': 0}
    
    def _count_voice_breaks(self, voiced_frames: np.ndarray) -> int:
        """Count voice breaks (transitions from voiced to unvoiced)"""
        if len(voiced_frames) < 2:
            return 0
            
        transitions = np.diff(voiced_frames.astype(int))
        voice_breaks = np.sum(transitions == -1)  # Voiced to unvoiced transitions
        return int(voice_breaks)