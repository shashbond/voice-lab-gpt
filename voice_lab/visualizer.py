"""
Visualization module for Voice Lab GPT
Creates comprehensive plots for voice analysis results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
from scipy import signal
import io
import base64

class VoiceVisualizer:
    """Creates comprehensive visualizations for voice analysis"""
    
    def __init__(self):
        """Initialize visualizer with plotting parameters"""
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Plot parameters
        self.figsize_large = (15, 10)
        self.figsize_medium = (12, 8)
        self.figsize_small = (8, 6)
        self.dpi = 150
        
    def create_comprehensive_visualization(self, 
                                         audio: np.ndarray,
                                         sr: int,
                                         acoustic_features: Dict[str, float],
                                         grbas_results: Dict[str, Any],
                                         spectrographic_features: Dict[str, Any],
                                         clinical_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Create comprehensive visualization suite
        
        Args:
            audio: Audio signal
            sr: Sample rate
            acoustic_features: Acoustic analysis results
            grbas_results: GRBAS estimation results
            spectrographic_features: Spectrographic analysis results
            clinical_results: Clinical scenario mapping results
            
        Returns:
            Dictionary of base64 encoded plot images
        """
        
        plots = {}
        
        try:
            # 1. Waveform and basic analysis
            plots['waveform'] = self._create_waveform_plot(audio, sr)
            
            # 2. Spectrogram analysis
            plots['spectrogram'] = self._create_spectrogram_plot(audio, sr, spectrographic_features)
            
            # 3. GRBAS visualization
            plots['grbas'] = self._create_grbas_visualization(grbas_results)
            
            # 4. Acoustic features dashboard
            plots['acoustic_dashboard'] = self._create_acoustic_dashboard(acoustic_features)
            
            # 5. Clinical summary visualization
            plots['clinical_summary'] = self._create_clinical_summary(clinical_results, grbas_results)
            
            # 6. Comprehensive dashboard
            plots['comprehensive_dashboard'] = self._create_comprehensive_dashboard(
                audio, sr, acoustic_features, grbas_results, spectrographic_features
            )
            
            # 7. F0 and periodicity analysis
            plots['f0_analysis'] = self._create_f0_analysis(audio, sr, acoustic_features)
            
            # 8. Spectral analysis
            plots['spectral_analysis'] = self._create_spectral_analysis(audio, sr, acoustic_features)
            
        except Exception as e:
            warnings.warn(f"Visualization creation failed: {str(e)}")
            plots['error'] = str(e)
        
        return plots
    
    def _create_waveform_plot(self, audio: np.ndarray, sr: int) -> str:
        """Create waveform visualization with amplitude envelope"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize_medium, sharex=True)
        
        time = np.linspace(0, len(audio) / sr, len(audio))
        
        # Waveform
        ax1.plot(time, audio, color='steelblue', alpha=0.8, linewidth=0.5)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Voice Waveform Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add envelope
        envelope = np.abs(librosa.stft(audio, hop_length=512))
        envelope_time = librosa.frames_to_time(range(envelope.shape[1]), sr=sr, hop_length=512)
        envelope_rms = np.sqrt(np.mean(envelope**2, axis=0))
        
        ax1.plot(envelope_time, envelope_rms, color='red', linewidth=2, alpha=0.7, label='RMS Envelope')
        ax1.plot(envelope_time, -envelope_rms, color='red', linewidth=2, alpha=0.7)
        ax1.legend()
        
        # Energy over time
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length, axis=0).T
        energy = np.sum(frames**2, axis=1)
        energy_time = np.arange(len(energy)) * hop_length / sr
        
        ax2.plot(energy_time, 10 * np.log10(energy + 1e-10), color='green', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Energy (dB)')
        ax2.set_title('Voice Energy Profile')
        ax2.grid(True, alpha=0.3)
        
        # Mark voice activity
        energy_threshold = np.percentile(energy, 30)
        voiced_segments = energy > energy_threshold
        
        for i, is_voiced in enumerate(voiced_segments):
            if is_voiced:
                ax2.axvspan(energy_time[i], energy_time[i] + hop_length/sr, 
                           alpha=0.2, color='yellow')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_spectrogram_plot(self, 
                                audio: np.ndarray, 
                                sr: int, 
                                spectrographic_features: Dict[str, Any]) -> str:
        """Create comprehensive spectrogram visualization"""
        
        fig = plt.figure(figsize=self.figsize_large)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1])
        
        # Main spectrogram
        ax_main = fig.add_subplot(gs[0, :])
        
        # Compute spectrogram
        n_fft = 1024
        hop_length = 256
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        db_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Display spectrogram
        img = librosa.display.specshow(db_spectrogram, sr=sr, hop_length=hop_length,
                                     x_axis='time', y_axis='hz', ax=ax_main,
                                     cmap='viridis', vmin=-80, vmax=0)
        
        ax_main.set_title('Voice Spectrogram Analysis', fontsize=14, fontweight='bold')
        ax_main.set_ylim(0, 8000)  # Focus on voice range
        
        # Add colorbar
        cbar = plt.colorbar(img, ax=ax_main)
        cbar.set_label('Magnitude (dB)', rotation=270, labelpad=15)
        
        # Overlay harmonic tracks if available
        if spectrographic_features.get('spectrogram_data'):
            self._overlay_harmonic_tracks(ax_main, audio, sr)
        
        # Add annotations for key findings
        self._add_spectrogram_annotations(ax_main, spectrographic_features)
        
        # Average spectrum
        ax_spectrum = fig.add_subplot(gs[1, 0])
        
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        avg_spectrum = np.mean(db_spectrogram, axis=1)
        
        ax_spectrum.plot(frequencies[:len(avg_spectrum)//2], avg_spectrum[:len(avg_spectrum)//2], 
                        color='blue', linewidth=2)
        ax_spectrum.set_xlabel('Frequency (Hz)')
        ax_spectrum.set_ylabel('Magnitude (dB)')
        ax_spectrum.set_title('Average Spectrum')
        ax_spectrum.set_xlim(0, 5000)
        ax_spectrum.grid(True, alpha=0.3)
        
        # Spectral characteristics box
        ax_info = fig.add_subplot(gs[1, 1])
        ax_info.axis('off')
        
        info_text = f"""Spectral Findings:
        
Harmonic Definition: {spectrographic_features.get('harmonic_definition', 'Unknown').title()}
Formant Clarity: {spectrographic_features.get('formant_clarity', 'Unknown').title()}
Breath Noise: {spectrographic_features.get('breath_noise_rating', 'Unknown').title()}
Aperiodicity: {spectrographic_features.get('overall_aperiodicity', 'Unknown').title()}
HF Decay: {spectrographic_features.get('hf_decay_pattern', 'Unknown').title()}

Voice Breaks: {spectrographic_features.get('voice_breaks_percentage', 0):.1f}%
Subharmonics: {'Present' if spectrographic_features.get('subharmonic_strength', 0) > 0.1 else 'Absent'}"""
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                    verticalalignment='top', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_grbas_visualization(self, grbas_results: Dict[str, Any]) -> str:
        """Create comprehensive GRBAS visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize_medium)
        
        # GRBAS bar chart
        dimensions = ['Grade', 'Roughness', 'Breathiness', 'Asthenia', 'Strain']
        ratings = [grbas_results[dim] for dim in ['G', 'R', 'B', 'A', 'S']]
        confidences = [grbas_results['confidence'][dim] for dim in ['G', 'R', 'B', 'A', 'S']]
        
        colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71']
        bars = ax1.bar(dimensions, ratings, color=colors, alpha=0.8)
        
        # Add confidence indicators
        for bar, rating, confidence in zip(bars, ratings, confidences):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{rating}\n({confidence:.0%})', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_ylim(0, 3.5)
        ax1.set_ylabel('GRBAS Rating (0-3)')
        ax1.set_title('GRBAS Perceptual Ratings', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # GRBAS radar chart
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the plot
        ratings_radar = ratings + [ratings[0]]
        
        ax2 = plt.subplot(2, 2, 2, projection='polar')
        ax2.plot(angles, ratings_radar, 'o-', linewidth=2, color='#3498db', alpha=0.8)
        ax2.fill(angles, ratings_radar, alpha=0.25, color='#3498db')
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(dimensions)
        ax2.set_ylim(0, 3)
        ax2.set_title('GRBAS Profile', pad=20, fontweight='bold')
        ax2.grid(True)
        
        # Confidence levels
        conf_values = list(confidences)
        conf_labels = ['G', 'R', 'B', 'A', 'S']
        
        wedges, texts, autotexts = ax3.pie(conf_values, labels=conf_labels, autopct='%1.0f%%',
                                         colors=colors, startangle=90)
        ax3.set_title('Confidence Levels', fontweight='bold')
        
        # Overall assessment
        ax4.axis('off')
        overall_score = np.mean(ratings)
        overall_confidence = grbas_results['confidence']['overall']
        
        # Create severity assessment
        if overall_score <= 0.5:
            severity = "Normal"
            severity_color = "green"
        elif overall_score <= 1.5:
            severity = "Mild"
            severity_color = "orange"
        elif overall_score <= 2.5:
            severity = "Moderate"
            severity_color = "red"
        else:
            severity = "Severe"
            severity_color = "darkred"
        
        assessment_text = f"""Overall Assessment:

Severity: {severity}
Average Score: {overall_score:.1f}/3.0
Overall Confidence: {overall_confidence:.0%}

Key Findings:
• Grade (G): {grbas_results['G']}/3
• Most Prominent: {dimensions[np.argmax(ratings)]}
• Confidence Range: {min(confidences):.0%} - {max(confidences):.0%}"""
        
        ax4.text(0.05, 0.95, assessment_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=severity_color, alpha=0.2))
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_acoustic_dashboard(self, acoustic_features: Dict[str, float]) -> str:
        """Create comprehensive acoustic features dashboard"""
        
        fig = plt.figure(figsize=self.figsize_large)
        gs = GridSpec(3, 3, figure=fig)
        
        # F0 measures
        ax1 = fig.add_subplot(gs[0, 0])
        f0_measures = ['Mean', 'Median', 'SD', 'Range']
        f0_values = [
            acoustic_features.get('f0_mean', 0),
            acoustic_features.get('f0_median', 0),
            acoustic_features.get('f0_std', 0),
            acoustic_features.get('f0_max', 0) - acoustic_features.get('f0_min', 0)
        ]
        
        bars1 = ax1.bar(f0_measures, f0_values, color='skyblue')
        ax1.set_title('F0 Characteristics (Hz)', fontweight='bold')
        ax1.set_ylabel('Frequency (Hz)')
        
        for bar, value in zip(bars1, f0_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(f0_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Perturbation measures
        ax2 = fig.add_subplot(gs[0, 1])
        pert_measures = ['Jitter (local)', 'Jitter (RAP)', 'Shimmer (local)', 'Shimmer (APQ11)']
        pert_values = [
            acoustic_features.get('jitter_local', 0) * 1000,  # Convert to per mille
            acoustic_features.get('jitter_rap', 0) * 1000,
            acoustic_features.get('shimmer_local', 0) * 100,  # Convert to percentage
            acoustic_features.get('shimmer_apq11', 0) * 100
        ]
        
        colors2 = ['lightcoral', 'coral', 'lightsalmon', 'sandybrown']
        bars2 = ax2.bar(pert_measures, pert_values, color=colors2)
        ax2.set_title('Perturbation Measures', fontweight='bold')
        ax2.set_ylabel('Value (‰ or %)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, pert_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(pert_values)*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Voice quality measures
        ax3 = fig.add_subplot(gs[0, 2])
        quality_measures = ['HNR (dB)', 'CPP (dB)', 'NHR']
        quality_values = [
            acoustic_features.get('hnr_mean', 0),
            acoustic_features.get('cpp_mean', 0),
            acoustic_features.get('nhr_mean', 0) * 100  # Scale NHR
        ]
        
        bars3 = ax3.bar(quality_measures, quality_values, color='lightgreen')
        ax3.set_title('Voice Quality Indices', fontweight='bold')
        ax3.set_ylabel('Value')
        
        for bar, value in zip(bars3, quality_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(quality_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Spectral measures
        ax4 = fig.add_subplot(gs[1, :])
        spectral_measures = ['Spectral Tilt', 'Spectral Centroid', 'High-Freq Ratio', 'Rolloff 95%']
        spectral_values = [
            acoustic_features.get('spectral_tilt', 0),
            acoustic_features.get('spectral_centroid', 0) / 1000,  # Convert to kHz
            acoustic_features.get('high_freq_ratio', 0) * 100,  # Convert to percentage
            acoustic_features.get('spectral_rolloff_95', 0) / 1000  # Convert to kHz
        ]
        
        colors4 = ['mediumpurple', 'mediumorchid', 'plum', 'thistle']
        bars4 = ax4.bar(spectral_measures, spectral_values, color=colors4)
        ax4.set_title('Spectral Characteristics', fontweight='bold')
        ax4.set_ylabel('Value')
        
        for bar, value in zip(bars4, spectral_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(spectral_values)*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Formant information
        ax5 = fig.add_subplot(gs[2, 0])
        if acoustic_features.get('f1_mean', 0) > 0:
            formant_freqs = ['F1', 'F2', 'F3']
            formant_values = [
                acoustic_features.get('f1_mean', 0),
                acoustic_features.get('f2_mean', 0),
                acoustic_features.get('f3_mean', 0)
            ]
            
            bars5 = ax5.bar(formant_freqs, formant_values, color='gold')
            ax5.set_title('Formant Frequencies (Hz)', fontweight='bold')
            ax5.set_ylabel('Frequency (Hz)')
            
            for bar, value in zip(bars5, formant_values):
                if value > 0:
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + max(formant_values)*0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=9)
        else:
            ax5.text(0.5, 0.5, 'Formant data\nnot available', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Formant Frequencies', fontweight='bold')
        
        # Intensity and dynamic measures
        ax6 = fig.add_subplot(gs[2, 1])
        intensity_measures = ['Mean Intensity', 'Intensity Range', 'Dynamic Range']
        intensity_values = [
            acoustic_features.get('intensity_mean', 0),
            acoustic_features.get('intensity_range', 0),
            acoustic_features.get('dynamic_range_db', 0)
        ]
        
        bars6 = ax6.bar(intensity_measures, intensity_values, color='lightblue')
        ax6.set_title('Intensity Measures (dB)', fontweight='bold')
        ax6.set_ylabel('Level (dB)')
        ax6.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars6, intensity_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + max(intensity_values)*0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Summary statistics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        voiced_fraction = acoustic_features.get('voiced_fraction', 0)
        pause_ratio = acoustic_features.get('pause_ratio', 0)
        voice_breaks = acoustic_features.get('voice_breaks', 0)
        
        summary_text = f"""Voice Characteristics:

Voiced Fraction: {voiced_fraction:.2f}
Pause Ratio: {pause_ratio:.2f}
Voice Breaks: {voice_breaks}

F0 Stability: {self._assess_f0_stability(acoustic_features)}
Periodicity: {self._assess_periodicity(acoustic_features)}
Voice Quality: {self._assess_voice_quality(acoustic_features)}"""
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_clinical_summary(self, 
                               clinical_results: Dict[str, Any], 
                               grbas_results: Dict[str, Any]) -> str:
        """Create clinical summary visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize_medium)
        
        # Primary impression
        ax1.axis('off')
        primary = clinical_results['primary_impression']
        
        impression_text = f"""PRIMARY IMPRESSION

Condition: {primary['condition'].replace('_', ' ').title()}

Confidence: {primary['confidence']:.0%}

Pattern Strength: {primary['pattern_strength']:.2f}

Description:
{primary['description']}"""
        
        # Color code by confidence
        if primary['confidence'] > 0.7:
            bg_color = 'lightgreen'
        elif primary['confidence'] > 0.4:
            bg_color = 'lightyellow'
        else:
            bg_color = 'lightcoral'
        
        ax1.text(0.05, 0.95, impression_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, alpha=0.7))
        
        # Differential diagnoses
        ax2.axis('off')
        if clinical_results.get('differential_diagnoses'):
            diff_text = "DIFFERENTIAL DIAGNOSES\n\n"
            for i, diff in enumerate(clinical_results['differential_diagnoses'][:4], 1):
                diff_text += f"{i}. {diff['condition'].replace('_', ' ').title()}\n"
                diff_text += f"   Confidence: {diff['confidence']:.0%}\n\n"
        else:
            diff_text = "DIFFERENTIAL DIAGNOSES\n\nNone identified above\nminimum threshold"
        
        ax2.text(0.05, 0.95, diff_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # Triage information
        ax3.axis('off')
        triage = clinical_results['triage']
        
        triage_colors = {
            'urgent': 'red',
            'soon': 'orange',
            'routine': 'green',
            'conservative': 'lightgreen'
        }
        
        triage_text = f"""TRIAGE RECOMMENDATION

Level: {triage['level'].upper()}
Timeframe: {triage['timeframe']}
Specialist: {triage['specialist']}

Reasoning:
{triage['reasoning']}"""
        
        ax3.text(0.05, 0.95, triage_text, transform=ax3.transAxes,
                verticalalignment='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", 
                         facecolor=triage_colors.get(triage['level'], 'gray'), 
                         alpha=0.3))
        
        # Pattern matching visualization
        if clinical_results.get('pattern_matches'):
            pattern_matches = clinical_results['pattern_matches']
            
            conditions = list(pattern_matches.keys())[:6]  # Top 6
            scores = [pattern_matches[cond]['confidence_weighted_score'] for cond in conditions]
            
            # Clean condition names
            clean_conditions = [cond.replace('_', ' ').title()[:15] + '...' if len(cond) > 15 
                              else cond.replace('_', ' ').title() for cond in conditions]
            
            bars = ax4.barh(clean_conditions, scores, color=plt.cm.viridis(np.linspace(0, 1, len(conditions))))
            ax4.set_xlabel('Pattern Match Score')
            ax4.set_title('Clinical Pattern Matching', fontweight='bold')
            ax4.set_xlim(0, 1)
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.2f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_comprehensive_dashboard(self, 
                                      audio: np.ndarray,
                                      sr: int,
                                      acoustic_features: Dict[str, float],
                                      grbas_results: Dict[str, Any],
                                      spectrographic_features: Dict[str, Any]) -> str:
        """Create comprehensive analysis dashboard"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Voice Lab GPT - Comprehensive Analysis Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # Waveform (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        time = np.linspace(0, len(audio) / sr, len(audio))
        ax1.plot(time, audio, color='steelblue', alpha=0.7, linewidth=0.5)
        ax1.set_title('Voice Waveform', fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        db_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        img = librosa.display.specshow(db_spec, sr=sr, hop_length=256,
                                     x_axis='time', y_axis='hz', ax=ax2,
                                     cmap='viridis', vmin=-60, vmax=0)
        ax2.set_title('Spectrogram', fontweight='bold')
        ax2.set_ylim(0, 4000)
        
        # GRBAS radar (middle left)
        ax3 = plt.subplot(gs[1, 0], projection='polar')
        
        dimensions = ['Grade', 'Roughness', 'Breathiness', 'Asthenia', 'Strain']
        ratings = [grbas_results[dim] for dim in ['G', 'R', 'B', 'A', 'S']]
        
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        ratings_radar = ratings + [ratings[0]]
        
        ax3.plot(angles, ratings_radar, 'o-', linewidth=2, color='red', alpha=0.8)
        ax3.fill(angles, ratings_radar, alpha=0.25, color='red')
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([d[:1] for d in dimensions])  # Use first letter only
        ax3.set_ylim(0, 3)
        ax3.set_title('GRBAS\nProfile', fontweight='bold')
        
        # Key acoustic measures (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        
        key_measures = ['F0', 'Jitter', 'Shimmer', 'HNR', 'CPP']
        key_values = [
            min(1, acoustic_features.get('f0_mean', 150) / 300),  # Normalize
            min(1, acoustic_features.get('jitter_local', 0) * 100),
            min(1, acoustic_features.get('shimmer_local', 0) * 20),
            min(1, max(0, acoustic_features.get('hnr_mean', 0) / 20)),
            min(1, max(0, acoustic_features.get('cpp_mean', 0) / 20))
        ]
        
        bars = ax4.bar(key_measures, key_values, color=['blue', 'red', 'orange', 'green', 'purple'])
        ax4.set_title('Key Acoustic\nMeasures', fontweight='bold')
        ax4.set_ylabel('Normalized Value')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        # Voice quality indicators (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        
        quality_labels = ['Stability', 'Clarity', 'Strength']
        quality_values = [
            max(0, min(100, 100 - acoustic_features.get('jitter_local', 0) * 5000)),
            max(0, min(100, acoustic_features.get('hnr_mean', 0) * 5)),
            max(0, min(100, (acoustic_features.get('intensity_mean', 60) - 40) * 2.5))
        ]
        
        wedges, texts, autotexts = ax5.pie(quality_values, labels=quality_labels, autopct='%1.0f%%',
                                         colors=['lightblue', 'lightgreen', 'lightyellow'])
        ax5.set_title('Voice Quality\nIndicators', fontweight='bold')
        
        # Clinical summary (middle far right)
        ax6 = fig.add_subplot(gs[1, 3])
        ax6.axis('off')
        
        # Overall severity
        avg_grbas = np.mean(ratings)
        if avg_grbas <= 0.5:
            severity = "Normal"
            sev_color = "green"
        elif avg_grbas <= 1.5:
            severity = "Mild"
            sev_color = "yellow"
        elif avg_grbas <= 2.5:
            severity = "Moderate" 
            sev_color = "orange"
        else:
            severity = "Severe"
            sev_color = "red"
        
        summary_text = f"""Clinical Summary:
        
Severity: {severity}
GRBAS: G{grbas_results['G']}R{grbas_results['R']}B{grbas_results['B']}A{grbas_results['A']}S{grbas_results['S']}

Key Findings:
• F0: {acoustic_features.get('f0_mean', 0):.0f} Hz
• HNR: {acoustic_features.get('hnr_mean', 0):.1f} dB
• Jitter: {acoustic_features.get('jitter_local', 0)*1000:.1f}‰

Confidence: {grbas_results['confidence']['overall']:.0%}"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=sev_color, alpha=0.3))
        
        # Spectral analysis (bottom left)
        ax7 = fig.add_subplot(gs[2, :2])
        
        # Compute average spectrum
        n_fft = 1024
        freqs = np.fft.fftfreq(n_fft, 1/sr)[:n_fft//2]
        spectrum = np.abs(np.fft.fft(audio * np.hanning(len(audio)), n_fft))[:n_fft//2]
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        ax7.plot(freqs, spectrum_db, color='purple', linewidth=1.5)
        ax7.set_xlabel('Frequency (Hz)')
        ax7.set_ylabel('Magnitude (dB)')
        ax7.set_title('Average Spectrum', fontweight='bold')
        ax7.set_xlim(0, 4000)
        ax7.grid(True, alpha=0.3)
        
        # Add spectral tilt line
        if acoustic_features.get('spectral_tilt'):
            tilt = acoustic_features['spectral_tilt']
            tilt_line = spectrum_db[0] + tilt * np.log10(freqs[1:] + 1)
            ax7.plot(freqs[1:], tilt_line, '--', color='red', alpha=0.7, 
                    label=f'Tilt: {tilt:.1f} dB/oct')
            ax7.legend()
        
        # Feature importance (bottom right)
        ax8 = fig.add_subplot(gs[2, 2:])
        
        # Create feature importance based on GRBAS contributions
        feature_names = ['HNR', 'Jitter', 'Shimmer', 'CPP', 'F0 Var', 'Intensity']
        feature_importance = [
            min(1, max(0, 1 - acoustic_features.get('hnr_mean', 15) / 20)),  # Low HNR = high importance
            min(1, acoustic_features.get('jitter_local', 0) * 100),
            min(1, acoustic_features.get('shimmer_local', 0) * 20),
            min(1, max(0, 1 - acoustic_features.get('cpp_mean', 10) / 15)),
            min(1, acoustic_features.get('f0_std', 0) / 20),
            min(1, max(0, (70 - acoustic_features.get('intensity_mean', 70)) / 30))
        ]
        
        # Sort by importance
        sorted_pairs = sorted(zip(feature_names, feature_importance), 
                            key=lambda x: x[1], reverse=True)
        sorted_names, sorted_importance = zip(*sorted_pairs)
        
        bars = ax8.barh(sorted_names, sorted_importance, 
                       color=plt.cm.Reds(np.array(sorted_importance)))
        ax8.set_xlabel('Clinical Relevance')
        ax8.set_title('Feature Importance for Voice Quality', fontweight='bold')
        ax8.set_xlim(0, 1)
        
        # Add value labels
        for bar, importance in zip(bars, sorted_importance):
            width = bar.get_width()
            if width > 0.01:
                ax8.text(width/2, bar.get_y() + bar.get_height()/2,
                        f'{importance:.2f}', ha='center', va='center', 
                        fontsize=8, color='white' if importance > 0.5 else 'black')
        
        return self._fig_to_base64(fig)
    
    def _create_f0_analysis(self, 
                           audio: np.ndarray, 
                           sr: int, 
                           acoustic_features: Dict[str, float]) -> str:
        """Create detailed F0 analysis visualization"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize_medium)
        
        # Extract F0 contour
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=60, fmax=400, sr=sr)
            times = librosa.frames_to_time(range(len(f0)), sr=sr)
            
            # F0 contour
            ax1.plot(times, f0, 'o-', color='blue', markersize=3, alpha=0.7)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('F0 (Hz)')
            ax1.set_title('F0 Contour', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add mean and std lines
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                f0_mean = np.mean(f0_clean)
                f0_std = np.std(f0_clean)
                ax1.axhline(f0_mean, color='red', linestyle='--', alpha=0.8, label=f'Mean: {f0_mean:.1f} Hz')
                ax1.axhline(f0_mean + f0_std, color='orange', linestyle=':', alpha=0.6, label=f'+1 SD: {f0_std:.1f} Hz')
                ax1.axhline(f0_mean - f0_std, color='orange', linestyle=':', alpha=0.6)
                ax1.legend()
            
            # F0 histogram
            if len(f0_clean) > 0:
                ax2.hist(f0_clean, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
                ax2.axvline(f0_mean, color='red', linestyle='--', alpha=0.8)
                ax2.set_xlabel('F0 (Hz)')
                ax2.set_ylabel('Count')
                ax2.set_title('F0 Distribution', fontweight='bold')
                ax2.grid(True, alpha=0.3)
            
            # F0 stability over time
            if len(f0_clean) > 10:
                # Calculate local F0 variability
                window_size = min(10, len(f0_clean) // 5)
                local_std = []
                local_times = []
                
                for i in range(window_size, len(f0_clean) - window_size, window_size//2):
                    window = f0_clean[i-window_size:i+window_size]
                    local_std.append(np.std(window))
                    local_times.append(times[i] if i < len(times) else times[-1])
                
                ax3.plot(local_times, local_std, 'o-', color='green')
                ax3.set_xlabel('Time (s)')
                ax3.set_ylabel('Local F0 Variability (Hz)')
                ax3.set_title('F0 Stability Over Time', fontweight='bold')
                ax3.grid(True, alpha=0.3)
            
        except Exception as e:
            ax1.text(0.5, 0.5, 'F0 extraction\nfailed', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'F0 data\nnot available', ha='center', va='center', transform=ax2.transAxes)
            ax3.text(0.5, 0.5, 'F0 stability\nanalysis failed', ha='center', va='center', transform=ax3.transAxes)
        
        # F0 statistics summary
        ax4.axis('off')
        
        f0_stats_text = f"""F0 Statistics:

Mean: {acoustic_features.get('f0_mean', 0):.1f} Hz
Median: {acoustic_features.get('f0_median', 0):.1f} Hz
SD: {acoustic_features.get('f0_std', 0):.1f} Hz
Range: {acoustic_features.get('f0_min', 0):.1f} - {acoustic_features.get('f0_max', 0):.1f} Hz

Voiced Fraction: {acoustic_features.get('voiced_fraction', 0):.2f}
Semitone SD: {acoustic_features.get('f0_semitone_std', 0):.2f}

Interpretation:
{self._interpret_f0_characteristics(acoustic_features)}"""
        
        ax4.text(0.05, 0.95, f0_stats_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_spectral_analysis(self, 
                                 audio: np.ndarray, 
                                 sr: int, 
                                 acoustic_features: Dict[str, float]) -> str:
        """Create detailed spectral analysis visualization"""
        
        fig = plt.figure(figsize=self.figsize_large)
        gs = GridSpec(2, 3, figure=fig)
        
        # Compute spectrum
        n_fft = 2048
        spectrum = np.abs(np.fft.fft(audio * np.hanning(len(audio)), n_fft))[:n_fft//2]
        freqs = np.fft.fftfreq(n_fft, 1/sr)[:n_fft//2]
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        # Full spectrum
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(freqs, spectrum_db, color='blue', linewidth=1)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Complete Frequency Spectrum', fontweight='bold')
        ax1.set_xlim(0, sr//2)
        ax1.grid(True, alpha=0.3)
        
        # Add spectral tilt line
        spectral_tilt = acoustic_features.get('spectral_tilt', 0)
        if spectral_tilt != 0:
            # Create tilt line
            ref_freq = 1000  # Reference frequency
            ref_idx = np.argmin(np.abs(freqs - ref_freq))
            ref_level = spectrum_db[ref_idx]
            
            tilt_line = ref_level + spectral_tilt * np.log10(freqs[1:] / ref_freq)
            ax1.plot(freqs[1:], tilt_line, '--', color='red', alpha=0.7, 
                    linewidth=2, label=f'Spectral Tilt: {spectral_tilt:.1f} dB/oct')
            ax1.legend()
        
        # Voice range spectrum (0-4000 Hz)
        ax2 = fig.add_subplot(gs[1, 0])
        voice_mask = freqs <= 4000
        ax2.plot(freqs[voice_mask], spectrum_db[voice_mask], color='green', linewidth=1.5)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title('Voice Range (0-4kHz)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Mark formant regions
        formant_regions = [(200, 1000, 'F1'), (800, 2500, 'F2'), (1500, 3500, 'F3')]
        colors = ['red', 'blue', 'orange']
        
        for (f_min, f_max, label), color in zip(formant_regions, colors):
            region_mask = (freqs >= f_min) & (freqs <= f_max) & (freqs <= 4000)
            if np.sum(region_mask) > 0:
                ax2.fill_between(freqs[region_mask], 
                               spectrum_db[region_mask], 
                               np.min(spectrum_db[voice_mask]),
                               alpha=0.2, color=color, label=label)
        ax2.legend()
        
        # High frequency analysis
        ax3 = fig.add_subplot(gs[1, 1])
        hf_mask = (freqs >= 2000) & (freqs <= 8000)
        if np.sum(hf_mask) > 0:
            ax3.plot(freqs[hf_mask], spectrum_db[hf_mask], color='purple', linewidth=1.5)
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Magnitude (dB)')
            ax3.set_title('High Frequency (2-8kHz)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Fit line to show HF decay
            log_freqs_hf = np.log10(freqs[hf_mask])
            spectrum_hf = spectrum_db[hf_mask]
            
            if len(log_freqs_hf) > 1 and len(spectrum_hf) > 1:
                z = np.polyfit(log_freqs_hf, spectrum_hf, 1)
                p = np.poly1d(z)
                ax3.plot(freqs[hf_mask], p(log_freqs_hf), '--', color='red', 
                        alpha=0.8, label=f'Decay: {z[0]:.1f} dB/oct')
                ax3.legend()
        
        # Spectral characteristics
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        
        # Calculate spectral characteristics
        spectral_centroid = acoustic_features.get('spectral_centroid', 0)
        spectral_rolloff = acoustic_features.get('spectral_rolloff_95', 0)
        high_freq_ratio = acoustic_features.get('high_freq_ratio', 0)
        
        spectral_text = f"""Spectral Characteristics:

Centroid: {spectral_centroid:.0f} Hz
Rolloff (95%): {spectral_rolloff:.0f} Hz
Tilt: {spectral_tilt:.1f} dB/oct
HF Ratio: {high_freq_ratio:.3f}

Energy Distribution:
0-1kHz: {self._calculate_energy_ratio(spectrum, freqs, 0, 1000):.1%}
1-2kHz: {self._calculate_energy_ratio(spectrum, freqs, 1000, 2000):.1%}
2-4kHz: {self._calculate_energy_ratio(spectrum, freqs, 2000, 4000):.1%}
4-8kHz: {self._calculate_energy_ratio(spectrum, freqs, 4000, 8000):.1%}

Interpretation:
{self._interpret_spectral_characteristics(acoustic_features)}"""
        
        ax4.text(0.05, 0.95, spectral_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.7))
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        return base64.b64encode(plot_data).decode('utf-8')
    
    def _overlay_harmonic_tracks(self, ax, audio: np.ndarray, sr: int):
        """Overlay harmonic tracks on spectrogram"""
        try:
            # Extract F0 track
            f0, _, _ = librosa.pyin(audio, fmin=60, fmax=400, sr=sr)
            times = librosa.frames_to_time(range(len(f0)), sr=sr)
            
            # Plot fundamental and harmonics
            for harmonic in [1, 2, 3, 4]:
                harmonic_track = f0 * harmonic
                valid_mask = ~np.isnan(harmonic_track) & (harmonic_track < 4000)
                
                if np.sum(valid_mask) > 0:
                    ax.plot(times[valid_mask], harmonic_track[valid_mask], 
                           color='red' if harmonic == 1 else 'yellow',
                           linewidth=2 if harmonic == 1 else 1,
                           alpha=0.8, label=f'H{harmonic}' if harmonic <= 2 else None)
            
            if np.sum(~np.isnan(f0)) > 0:
                ax.legend(loc='upper right')
                
        except Exception:
            pass  # Fail silently if harmonic tracking fails
    
    def _add_spectrogram_annotations(self, ax, spectrographic_features: Dict[str, Any]):
        """Add annotations to spectrogram based on findings"""
        
        # Add text annotations for key findings
        findings = []
        
        if spectrographic_features.get('harmonic_definition') == 'poor':
            findings.append("Poor harmonic definition")
        
        if spectrographic_features.get('subharmonic_strength', 0) > 0.1:
            findings.append("Subharmonics present")
        
        if spectrographic_features.get('breath_noise_rating') in ['moderate', 'severe']:
            findings.append("Elevated breath noise")
        
        if spectrographic_features.get('voice_breaks_percentage', 0) > 10:
            findings.append("Frequent voice breaks")
        
        # Add findings as text box
        if findings:
            findings_text = "Findings:\n" + "\n".join([f"• {finding}" for finding in findings])
            ax.text(0.02, 0.98, findings_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _assess_f0_stability(self, acoustic_features: Dict[str, float]) -> str:
        """Assess F0 stability from features"""
        f0_std = acoustic_features.get('f0_std', 0)
        
        if f0_std <= 5:
            return "Excellent stability"
        elif f0_std <= 10:
            return "Good stability"
        elif f0_std <= 20:
            return "Moderate variability"
        else:
            return "High variability"
    
    def _assess_periodicity(self, acoustic_features: Dict[str, float]) -> str:
        """Assess periodicity from jitter/shimmer"""
        jitter = acoustic_features.get('jitter_local', 0)
        shimmer = acoustic_features.get('shimmer_local', 0)
        
        if jitter <= 0.01 and shimmer <= 0.05:
            return "Excellent"
        elif jitter <= 0.02 and shimmer <= 0.08:
            return "Good" 
        elif jitter <= 0.04 and shimmer <= 0.12:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_voice_quality(self, acoustic_features: Dict[str, float]) -> str:
        """Assess overall voice quality from HNR/CPP"""
        hnr = acoustic_features.get('hnr_mean', 0)
        cpp = acoustic_features.get('cpp_mean', 0)
        
        if hnr >= 15 and cpp >= 15:
            return "Excellent"
        elif hnr >= 10 and cpp >= 10:
            return "Good"
        elif hnr >= 5 and cpp >= 5:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_energy_ratio(self, spectrum: np.ndarray, freqs: np.ndarray, 
                               f_min: float, f_max: float) -> float:
        """Calculate energy ratio in frequency band"""
        mask = (freqs >= f_min) & (freqs <= f_max)
        if np.sum(mask) == 0:
            return 0.0
        
        band_energy = np.sum(spectrum[mask]**2)
        total_energy = np.sum(spectrum**2)
        
        return band_energy / (total_energy + 1e-10)
    
    def _interpret_f0_characteristics(self, acoustic_features: Dict[str, float]) -> str:
        """Interpret F0 characteristics"""
        f0_mean = acoustic_features.get('f0_mean', 0)
        f0_std = acoustic_features.get('f0_std', 0)
        voiced_fraction = acoustic_features.get('voiced_fraction', 0)
        
        interpretations = []
        
        if f0_mean < 100:
            interpretations.append("Low fundamental frequency")
        elif f0_mean > 250:
            interpretations.append("High fundamental frequency")
        else:
            interpretations.append("Normal frequency range")
        
        if f0_std > 20:
            interpretations.append("High F0 variability")
        elif f0_std < 5:
            interpretations.append("Monotone tendency")
        
        if voiced_fraction < 0.5:
            interpretations.append("Reduced voicing")
        
        return "; ".join(interpretations) if interpretations else "Normal F0 characteristics"
    
    def _interpret_spectral_characteristics(self, acoustic_features: Dict[str, float]) -> str:
        """Interpret spectral characteristics"""
        spectral_tilt = acoustic_features.get('spectral_tilt', 0)
        high_freq_ratio = acoustic_features.get('high_freq_ratio', 0)
        
        interpretations = []
        
        if spectral_tilt < -15:
            interpretations.append("Steep spectral slope (breathy)")
        elif spectral_tilt > -5:
            interpretations.append("Flat spectral slope (tense)")
        else:
            interpretations.append("Normal spectral slope")
        
        if high_freq_ratio > 0.3:
            interpretations.append("Excessive high-frequency energy")
        elif high_freq_ratio < 0.05:
            interpretations.append("Reduced high-frequency content")
        
        return "; ".join(interpretations) if interpretations else "Normal spectral characteristics"