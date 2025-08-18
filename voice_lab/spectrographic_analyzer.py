"""
Spectrographic analysis for Voice Lab GPT
Provides detailed visual and quantitative spectrographic findings
"""

import numpy as np
import librosa
import scipy.signal
from scipy import ndimage
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings

class SpectrographicAnalyzer:
    """Analyzes spectrograms and extracts clinical findings"""
    
    def __init__(self, sr: int = 16000):
        """
        Initialize spectrographic analyzer
        
        Args:
            sr: Sample rate
        """
        self.sr = sr
        
    def analyze_spectrogram(self, 
                          audio: np.ndarray,
                          window_ms: float = 5.0,
                          overlap_ratio: float = 0.8) -> Dict[str, any]:
        """
        Comprehensive spectrographic analysis
        
        Args:
            audio: Audio signal
            window_ms: Window length in milliseconds for spectrogram
            overlap_ratio: Overlap ratio for STFT
            
        Returns:
            Dictionary of spectrographic findings
        """
        # Calculate STFT parameters
        window_length = int(window_ms * self.sr / 1000)
        hop_length = int(window_length * (1 - overlap_ratio))
        
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=window_length*2, hop_length=hop_length, 
                           window='hanning', center=True)
        magnitude = np.abs(stft)
        power_spectrogram = magnitude ** 2
        
        # Convert to dB
        db_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # Frequency and time axes
        frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=window_length*2)
        times = librosa.frames_to_time(range(stft.shape[1]), sr=self.sr, 
                                     hop_length=hop_length)
        
        findings = {}
        
        # Harmonic structure analysis
        harmonic_findings = self._analyze_harmonic_structure(
            db_spectrogram, frequencies, times, audio
        )
        findings.update(harmonic_findings)
        
        # Noise analysis
        noise_findings = self._analyze_noise_patterns(
            db_spectrogram, frequencies, times
        )
        findings.update(noise_findings)
        
        # Aperiodicity analysis
        aperiodicity_findings = self._analyze_aperiodicity(
            db_spectrogram, power_spectrogram, frequencies, times, audio
        )
        findings.update(aperiodicity_findings)
        
        # High-frequency analysis
        hf_findings = self._analyze_high_frequency_content(
            db_spectrogram, frequencies, times
        )
        findings.update(hf_findings)
        
        # Dynamic characteristics
        dynamic_findings = self._analyze_dynamic_characteristics(
            db_spectrogram, frequencies, times
        )
        findings.update(dynamic_findings)
        
        # Store spectrogram data for visualization
        findings['spectrogram_data'] = {
            'magnitude_db': db_spectrogram,
            'frequencies': frequencies,
            'times': times,
            'power_spectrogram': power_spectrogram
        }
        
        return findings
    
    def _analyze_harmonic_structure(self, 
                                  db_spectrogram: np.ndarray,
                                  frequencies: np.ndarray,
                                  times: np.ndarray,
                                  audio: np.ndarray) -> Dict[str, any]:
        """Analyze harmonic structure and definition"""
        findings = {}
        
        try:
            # Estimate F0 for harmonic tracking
            f0_track = self._estimate_f0_track(audio)
            
            if len(f0_track) == 0 or np.all(f0_track == 0):
                return {
                    'harmonic_definition': 'poor',
                    'harmonic_clarity_score': 0.0,
                    'harmonic_energy_ratio': 0.0,
                    'formant_clarity': 'poor'
                }
            
            # Analyze harmonic clarity
            harmonic_scores = []
            harmonic_energies = []
            
            for i, f0 in enumerate(f0_track):
                if f0 > 0 and i < db_spectrogram.shape[1]:
                    time_slice = db_spectrogram[:, i]
                    
                    # Find harmonics (up to 5th harmonic)
                    harmonic_freqs = [f0 * h for h in range(1, 6) if f0 * h < self.sr/2]
                    harmonic_clarity = 0
                    harmonic_energy = 0
                    
                    for h_freq in harmonic_freqs:
                        # Find frequency bin closest to harmonic
                        freq_idx = np.argmin(np.abs(frequencies - h_freq))
                        
                        # Check harmonic peak vs surrounding noise
                        window = 3  # bins around harmonic
                        start_idx = max(0, freq_idx - window)
                        end_idx = min(len(time_slice), freq_idx + window + 1)
                        
                        if start_idx < end_idx:
                            peak_energy = time_slice[freq_idx]
                            surrounding_energy = np.mean(np.concatenate([
                                time_slice[start_idx:freq_idx],
                                time_slice[freq_idx+1:end_idx]
                            ]))
                            
                            harmonic_clarity += peak_energy - surrounding_energy
                            harmonic_energy += peak_energy
                    
                    harmonic_scores.append(harmonic_clarity)
                    harmonic_energies.append(harmonic_energy)
            
            if harmonic_scores:
                mean_clarity = np.mean(harmonic_scores)
                mean_energy = np.mean(harmonic_energies)
                
                # Classify harmonic definition
                if mean_clarity > 10:
                    definition = 'excellent'
                elif mean_clarity > 5:
                    definition = 'good'
                elif mean_clarity > 2:
                    definition = 'fair'
                else:
                    definition = 'poor'
                
                findings['harmonic_definition'] = definition
                findings['harmonic_clarity_score'] = float(mean_clarity)
                findings['harmonic_energy_ratio'] = float(mean_energy / np.max(db_spectrogram))
            else:
                findings['harmonic_definition'] = 'poor'
                findings['harmonic_clarity_score'] = 0.0
                findings['harmonic_energy_ratio'] = 0.0
            
            # Formant analysis from spectrogram
            formant_clarity = self._analyze_formant_clarity(db_spectrogram, frequencies)
            findings['formant_clarity'] = formant_clarity
            
        except Exception as e:
            warnings.warn(f"Harmonic structure analysis failed: {str(e)}")
            findings.update({
                'harmonic_definition': 'unknown',
                'harmonic_clarity_score': 0.0,
                'harmonic_energy_ratio': 0.0,
                'formant_clarity': 'unknown'
            })
        
        return findings
    
    def _analyze_noise_patterns(self, 
                              db_spectrogram: np.ndarray,
                              frequencies: np.ndarray,
                              times: np.ndarray) -> Dict[str, any]:
        """Analyze noise patterns in spectrogram"""
        findings = {}
        
        try:
            # Identify noise bands
            # High-frequency noise (above 4000 Hz)
            hf_idx = frequencies >= 4000
            if np.sum(hf_idx) > 0:
                hf_noise_level = np.mean(db_spectrogram[hf_idx, :])
                total_energy = np.mean(db_spectrogram)
                hf_noise_ratio = hf_noise_level / total_energy
                
                findings['high_freq_noise_level'] = float(hf_noise_level)
                findings['high_freq_noise_ratio'] = float(hf_noise_ratio)
            else:
                findings['high_freq_noise_level'] = -60.0
                findings['high_freq_noise_ratio'] = 0.0
            
            # Breath noise band (typically 1500-4000 Hz)
            breath_band_idx = (frequencies >= 1500) & (frequencies <= 4000)
            if np.sum(breath_band_idx) > 0:
                breath_noise_level = np.mean(db_spectrogram[breath_band_idx, :])
                findings['breath_noise_level'] = float(breath_noise_level)
                
                # Classify breath noise
                if breath_noise_level > -20:
                    breath_noise_rating = 'severe'
                elif breath_noise_level > -30:
                    breath_noise_rating = 'moderate'
                elif breath_noise_level > -40:
                    breath_noise_rating = 'mild'
                else:
                    breath_noise_rating = 'minimal'
                    
                findings['breath_noise_rating'] = breath_noise_rating
            else:
                findings['breath_noise_level'] = -60.0
                findings['breath_noise_rating'] = 'minimal'
            
            # Turbulent noise detection
            turbulence_score = self._detect_turbulent_noise(db_spectrogram, frequencies)
            findings['turbulent_noise_score'] = turbulence_score
            
            # Overall noise floor estimation
            noise_floor = np.percentile(db_spectrogram, 10)  # Bottom 10% as noise estimate
            findings['noise_floor_db'] = float(noise_floor)
            
        except Exception as e:
            warnings.warn(f"Noise pattern analysis failed: {str(e)}")
            findings.update({
                'high_freq_noise_level': -60.0,
                'high_freq_noise_ratio': 0.0,
                'breath_noise_level': -60.0,
                'breath_noise_rating': 'unknown',
                'turbulent_noise_score': 0.0,
                'noise_floor_db': -60.0
            })
        
        return findings
    
    def _analyze_aperiodicity(self, 
                            db_spectrogram: np.ndarray,
                            power_spectrogram: np.ndarray,
                            frequencies: np.ndarray,
                            times: np.ndarray,
                            audio: np.ndarray) -> Dict[str, any]:
        """Analyze aperiodic components and subharmonics"""
        findings = {}
        
        try:
            # Subharmonic detection
            subharmonic_presence = self._detect_subharmonics(db_spectrogram, frequencies, audio)
            findings.update(subharmonic_presence)
            
            # Sidebands detection
            sideband_presence = self._detect_sidebands(db_spectrogram, frequencies)
            findings['sideband_presence'] = sideband_presence
            
            # Aperiodicity measure using spectral irregularity
            aperiodicity_score = self._calculate_aperiodicity_score(power_spectrogram, frequencies)
            findings['aperiodicity_score'] = aperiodicity_score
            
            # Classify overall aperiodicity
            if subharmonic_presence.get('subharmonic_strength', 0) > 0.3 or aperiodicity_score > 0.7:
                aperiodicity_level = 'severe'
            elif subharmonic_presence.get('subharmonic_strength', 0) > 0.15 or aperiodicity_score > 0.4:
                aperiodicity_level = 'moderate'
            elif subharmonic_presence.get('subharmonic_strength', 0) > 0.05 or aperiodicity_score > 0.2:
                aperiodicity_level = 'mild'
            else:
                aperiodicity_level = 'minimal'
                
            findings['overall_aperiodicity'] = aperiodicity_level
            
        except Exception as e:
            warnings.warn(f"Aperiodicity analysis failed: {str(e)}")
            findings.update({
                'subharmonic_strength': 0.0,
                'subharmonic_frequency': 0.0,
                'sideband_presence': 'none',
                'aperiodicity_score': 0.0,
                'overall_aperiodicity': 'unknown'
            })
        
        return findings
    
    def _analyze_high_frequency_content(self, 
                                      db_spectrogram: np.ndarray,
                                      frequencies: np.ndarray,
                                      times: np.ndarray) -> Dict[str, any]:
        """Analyze high-frequency harmonic decay"""
        findings = {}
        
        try:
            # Analyze spectral slope in high frequencies
            hf_start = 2000  # Hz
            hf_end = min(8000, self.sr // 2)  # Hz
            
            hf_idx = (frequencies >= hf_start) & (frequencies <= hf_end)
            
            if np.sum(hf_idx) > 10:  # Need enough frequency bins
                hf_freqs = frequencies[hf_idx]
                hf_spectrum = np.mean(db_spectrogram[hf_idx, :], axis=1)  # Average over time
                
                # Linear regression on log frequency vs dB
                log_freqs = np.log10(hf_freqs)
                slope, intercept, r_value, _, _ = scipy.stats.linregress(log_freqs, hf_spectrum)
                
                findings['hf_decay_slope'] = float(slope)
                findings['hf_decay_r2'] = float(r_value ** 2)
                
                # Classify decay pattern
                if slope < -20:
                    decay_pattern = 'steep'
                elif slope < -10:
                    decay_pattern = 'moderate'
                elif slope < -5:
                    decay_pattern = 'gradual'
                else:
                    decay_pattern = 'flat'
                    
                findings['hf_decay_pattern'] = decay_pattern
                
                # Energy above 4000 Hz
                very_hf_idx = frequencies >= 4000
                if np.sum(very_hf_idx) > 0:
                    very_hf_energy = np.mean(db_spectrogram[very_hf_idx, :])
                    total_energy = np.mean(db_spectrogram)
                    very_hf_ratio = very_hf_energy / total_energy
                    findings['very_hf_energy_ratio'] = float(very_hf_ratio)
                else:
                    findings['very_hf_energy_ratio'] = 0.0
                    
            else:
                findings.update({
                    'hf_decay_slope': 0.0,
                    'hf_decay_r2': 0.0,
                    'hf_decay_pattern': 'unknown',
                    'very_hf_energy_ratio': 0.0
                })
                
        except Exception as e:
            warnings.warn(f"High-frequency analysis failed: {str(e)}")
            findings.update({
                'hf_decay_slope': 0.0,
                'hf_decay_r2': 0.0,
                'hf_decay_pattern': 'unknown',
                'very_hf_energy_ratio': 0.0
            })
        
        return findings
    
    def _analyze_dynamic_characteristics(self, 
                                       db_spectrogram: np.ndarray,
                                       frequencies: np.ndarray,
                                       times: np.ndarray) -> Dict[str, any]:
        """Analyze temporal dynamics in spectrogram"""
        findings = {}
        
        try:
            # Spectral variability over time
            spectral_std = np.std(db_spectrogram, axis=1)  # Variability across time for each frequency
            mean_spectral_variability = np.mean(spectral_std)
            
            # Temporal modulations
            temporal_std = np.std(db_spectrogram, axis=0)  # Variability across frequency for each time
            mean_temporal_variability = np.mean(temporal_std)
            
            findings['spectral_variability'] = float(mean_spectral_variability)
            findings['temporal_variability'] = float(mean_temporal_variability)
            
            # Voice breaks detection from spectrogram
            # Look for sudden drops in energy
            overall_energy = np.mean(db_spectrogram, axis=0)
            energy_threshold = np.mean(overall_energy) - 2 * np.std(overall_energy)
            
            voice_breaks = np.sum(overall_energy < energy_threshold)
            break_percentage = (voice_breaks / len(overall_energy)) * 100
            
            findings['voice_breaks_count'] = int(voice_breaks)
            findings['voice_breaks_percentage'] = float(break_percentage)
            
            # Tremor detection (look for regular modulations)
            tremor_strength = self._detect_tremor_from_spectrogram(db_spectrogram, times)
            findings['tremor_strength'] = tremor_strength
            
        except Exception as e:
            warnings.warn(f"Dynamic characteristics analysis failed: {str(e)}")
            findings.update({
                'spectral_variability': 0.0,
                'temporal_variability': 0.0,
                'voice_breaks_count': 0,
                'voice_breaks_percentage': 0.0,
                'tremor_strength': 0.0
            })
        
        return findings
    
    def _estimate_f0_track(self, audio: np.ndarray) -> np.ndarray:
        """Estimate F0 track using librosa"""
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, fmin=60, fmax=400, sr=self.sr, 
                frame_length=1024, hop_length=256
            )
            
            # Replace NaN with 0
            f0 = np.nan_to_num(f0, nan=0.0)
            
            return f0[voiced_flag]  # Return only voiced frames
            
        except Exception:
            return np.array([])
    
    def _analyze_formant_clarity(self, db_spectrogram: np.ndarray, frequencies: np.ndarray) -> str:
        """Analyze formant clarity from spectrogram"""
        try:
            # Look for clear formant bands in typical ranges
            f1_range = (200, 1000)
            f2_range = (800, 2500)
            f3_range = (1500, 3500)
            
            formant_clarity_scores = []
            
            for f_range in [f1_range, f2_range, f3_range]:
                freq_mask = (frequencies >= f_range[0]) & (frequencies <= f_range[1])
                if np.sum(freq_mask) > 0:
                    formant_region = db_spectrogram[freq_mask, :]
                    # Look for consistent energy peaks
                    mean_energy = np.mean(formant_region, axis=1)
                    peak_prominence = np.max(mean_energy) - np.mean(mean_energy)
                    formant_clarity_scores.append(peak_prominence)
            
            if formant_clarity_scores:
                mean_clarity = np.mean(formant_clarity_scores)
                if mean_clarity > 15:
                    return 'excellent'
                elif mean_clarity > 10:
                    return 'good'
                elif mean_clarity > 5:
                    return 'fair'
                else:
                    return 'poor'
            else:
                return 'poor'
                
        except Exception:
            return 'unknown'
    
    def _detect_turbulent_noise(self, db_spectrogram: np.ndarray, frequencies: np.ndarray) -> float:
        """Detect turbulent noise patterns"""
        try:
            # Turbulent noise typically appears as broadband noise with irregular patterns
            # Look for high-frequency irregularity
            hf_idx = frequencies >= 2000
            if np.sum(hf_idx) > 0:
                hf_region = db_spectrogram[hf_idx, :]
                
                # Calculate spectral irregularity
                spectral_diff = np.diff(hf_region, axis=0)
                irregularity = np.mean(np.abs(spectral_diff))
                
                # Normalize to 0-1 scale
                turbulence_score = min(irregularity / 10.0, 1.0)
                return float(turbulence_score)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _detect_subharmonics(self, db_spectrogram: np.ndarray, frequencies: np.ndarray, audio: np.ndarray) -> Dict[str, float]:
        """Detect subharmonic components"""
        try:
            f0_track = self._estimate_f0_track(audio)
            
            if len(f0_track) == 0:
                return {'subharmonic_strength': 0.0, 'subharmonic_frequency': 0.0}
            
            mean_f0 = np.mean(f0_track[f0_track > 0])
            
            # Look for energy at F0/2, F0/3, etc.
            subharmonic_strengths = []
            subharmonic_freqs = []
            
            for divisor in [2, 3, 4]:
                subh_freq = mean_f0 / divisor
                if subh_freq > frequencies[0]:  # Within analysis range
                    freq_idx = np.argmin(np.abs(frequencies - subh_freq))
                    
                    # Average energy at subharmonic frequency across time
                    subh_energy = np.mean(db_spectrogram[freq_idx, :])
                    
                    # Compare to surrounding frequencies
                    window = 2
                    start_idx = max(0, freq_idx - window)
                    end_idx = min(len(frequencies), freq_idx + window + 1)
                    surrounding_energy = np.mean(db_spectrogram[start_idx:end_idx, :])
                    
                    if surrounding_energy > 0:
                        strength = (subh_energy - surrounding_energy) / abs(surrounding_energy)
                        subharmonic_strengths.append(max(0, strength))
                        subharmonic_freqs.append(subh_freq)
            
            if subharmonic_strengths:
                max_idx = np.argmax(subharmonic_strengths)
                return {
                    'subharmonic_strength': float(subharmonic_strengths[max_idx]),
                    'subharmonic_frequency': float(subharmonic_freqs[max_idx])
                }
            else:
                return {'subharmonic_strength': 0.0, 'subharmonic_frequency': 0.0}
                
        except Exception:
            return {'subharmonic_strength': 0.0, 'subharmonic_frequency': 0.0}
    
    def _detect_sidebands(self, db_spectrogram: np.ndarray, frequencies: np.ndarray) -> str:
        """Detect sideband presence around harmonics"""
        try:
            # Look for modulation sidebands around harmonic frequencies
            # This is a simplified detection
            
            # Calculate spectral modulation index
            freq_resolution = frequencies[1] - frequencies[0]
            modulation_range = int(50 / freq_resolution)  # ±50 Hz around harmonics
            
            if modulation_range < 2:
                return 'none'
            
            # Look for periodic modulations in the spectrogram
            spectral_variance = np.var(db_spectrogram, axis=1)
            mean_variance = np.mean(spectral_variance)
            
            # High variance indicates potential sidebands
            if mean_variance > 100:  # Threshold for significant modulation
                return 'present'
            elif mean_variance > 50:
                return 'mild'
            else:
                return 'none'
                
        except Exception:
            return 'unknown'
    
    def _calculate_aperiodicity_score(self, power_spectrogram: np.ndarray, frequencies: np.ndarray) -> float:
        """Calculate overall aperiodicity score"""
        try:
            # Aperiodicity can be estimated by spectral irregularity
            # and temporal inconsistency
            
            # Spectral irregularity: variation in spectral shape
            mean_spectrum = np.mean(power_spectrogram, axis=1)
            spectral_profiles = power_spectrogram.T  # Each column is a time frame
            
            irregularity_scores = []
            for profile in spectral_profiles:
                if np.sum(profile) > 0:
                    normalized_profile = profile / np.sum(profile)
                    normalized_mean = mean_spectrum / np.sum(mean_spectrum)
                    
                    # Calculate Kullback-Leibler divergence as irregularity measure
                    kl_div = np.sum(normalized_profile * np.log(
                        (normalized_profile + 1e-10) / (normalized_mean + 1e-10)
                    ))
                    irregularity_scores.append(kl_div)
            
            if irregularity_scores:
                mean_irregularity = np.mean(irregularity_scores)
                # Normalize to 0-1 scale
                aperiodicity_score = min(mean_irregularity / 2.0, 1.0)
                return float(aperiodicity_score)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _detect_tremor_from_spectrogram(self, db_spectrogram: np.ndarray, times: np.ndarray) -> float:
        """Detect tremor from spectral modulations"""
        try:
            if len(times) < 10:
                return 0.0
            
            # Calculate overall energy over time
            energy_over_time = np.mean(db_spectrogram, axis=0)
            
            # Look for regular modulations (tremor) in 4-12 Hz range
            if len(energy_over_time) > 20:
                # Simple autocorrelation-based tremor detection
                autocorr = np.correlate(energy_over_time, energy_over_time, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Look for peaks in expected tremor range
                time_resolution = times[1] - times[0] if len(times) > 1 else 0.01
                tremor_lags = np.arange(int(1/(12*time_resolution)), int(1/(4*time_resolution)))
                
                if len(tremor_lags) > 0 and max(tremor_lags) < len(autocorr):
                    tremor_strength = np.max(autocorr[tremor_lags]) / autocorr[0]
                    return min(float(tremor_strength), 1.0)
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0