"""
Main Voice Lab GPT interface
Coordinates all analysis components and provides unified API
"""

import os
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import warnings
import traceback

from .audio_processor import AudioProcessor
from .acoustic_analyzer import AcousticAnalyzer
from .spectrographic_analyzer import SpectrographicAnalyzer
from .grbas_estimator import GRBASEstimator
from .clinical_mapper import ClinicalMapper
from .report_generator import ReportGenerator
from .visualizer import VoiceVisualizer

class VoiceLabGPT:
    """
    Main Voice Lab GPT class - Professional voice and speech analysis system
    
    This class coordinates all analysis components to provide comprehensive
    voice analysis with acoustic measures, perceptual estimates, clinical 
    impressions, and detailed reporting.
    """
    
    def __init__(self, 
                 sr: int = 16000,
                 target_lufs: float = -23.0,
                 silence_threshold_db: float = -40.0,
                 enable_visualizations: bool = True):
        """
        Initialize Voice Lab GPT
        
        Args:
            sr: Target sampling rate for analysis
            target_lufs: Target loudness normalization level
            silence_threshold_db: Threshold for silence removal
            enable_visualizations: Whether to generate visualization plots
        """
        
        self.sr = sr
        self.enable_visualizations = enable_visualizations
        
        # Initialize all analysis components
        try:
            self.audio_processor = AudioProcessor(sr, target_lufs, silence_threshold_db)
            self.acoustic_analyzer = AcousticAnalyzer(sr)
            self.spectrographic_analyzer = SpectrographicAnalyzer(sr)
            self.grbas_estimator = GRBASEstimator()
            self.clinical_mapper = ClinicalMapper()
            self.report_generator = ReportGenerator()
            
            if enable_visualizations:
                self.visualizer = VoiceVisualizer()
            else:
                self.visualizer = None
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Voice Lab GPT components: {str(e)}")
        
        # Analysis metadata
        self.version = "1.0.0"
        self.last_analysis = None
    
    def analyze_file(self, 
                    file_path: str,
                    voice_type: Optional[str] = None,
                    patient_info: Optional[str] = None,
                    task_description: Optional[str] = None,
                    generate_reports: bool = True,
                    generate_visualizations: bool = None) -> Dict[str, Any]:
        """
        Complete voice analysis pipeline for audio file
        
        Args:
            file_path: Path to audio file
            voice_type: 'male', 'female', or None for auto-detection
            patient_info: Patient identifier or description
            task_description: Task description (e.g., 'sustained /a/', 'reading passage')
            generate_reports: Whether to generate formatted reports
            generate_visualizations: Whether to generate plots (overrides instance setting)
            
        Returns:
            Comprehensive analysis results dictionary
        """
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Override visualization setting if specified
        if generate_visualizations is None:
            generate_visualizations = self.enable_visualizations
        
        try:
            # Start analysis timestamp
            analysis_start = datetime.datetime.now()
            
            # Step 1: Audio preprocessing
            print("🎵 Loading and preprocessing audio...")
            audio, processing_info = self.audio_processor.process_file(file_path)
            
            # Validate audio
            if len(audio) == 0:
                raise ValueError("Audio file is empty or could not be processed")
            
            if processing_info['processing_info']['final_duration'] < 1.0:
                warnings.warn("Audio duration is very short (<1s). Results may be unreliable.")
            
            # Step 2: Acoustic feature extraction
            print("🔬 Extracting acoustic features...")
            acoustic_features = self.acoustic_analyzer.extract_all_features(
                audio, voice_type=voice_type
            )
            
            # Step 3: Spectrographic analysis
            print("📊 Analyzing spectrogram...")
            spectrographic_features = self.spectrographic_analyzer.analyze_spectrogram(audio)
            
            # Step 4: GRBAS perceptual estimation
            print("👂 Estimating perceptual ratings...")
            grbas_results = self.grbas_estimator.estimate_grbas(
                acoustic_features, spectrographic_features
            )
            
            # Step 5: Clinical scenario mapping
            print("🏥 Mapping clinical scenarios...")
            clinical_results = self.clinical_mapper.analyze_clinical_scenarios(
                acoustic_features, grbas_results, spectrographic_features
            )
            
            # Step 6: Generate reports
            reports = None
            if generate_reports:
                print("📋 Generating reports...")
                reports = self.report_generator.generate_complete_report(
                    acoustic_features, grbas_results, spectrographic_features, 
                    clinical_results, patient_info, include_plots=generate_visualizations
                )
            
            # Step 7: Generate visualizations
            visualizations = None
            if generate_visualizations and self.visualizer:
                print("📈 Creating visualizations...")
                visualizations = self.visualizer.create_comprehensive_visualization(
                    audio, self.sr, acoustic_features, grbas_results,
                    spectrographic_features, clinical_results
                )
            
            # Analysis completion
            analysis_end = datetime.datetime.now()
            analysis_duration = (analysis_end - analysis_start).total_seconds()
            
            # Compile comprehensive results
            results = {
                'metadata': {
                    'file_path': file_path,
                    'patient_info': patient_info or os.path.basename(file_path),
                    'task_description': task_description or 'Voice analysis',
                    'voice_type': voice_type,
                    'analysis_timestamp': analysis_end.isoformat(),
                    'analysis_duration_seconds': analysis_duration,
                    'voice_lab_version': self.version,
                    'audio_duration_seconds': processing_info['processing_info']['final_duration']
                },
                'audio_processing': processing_info,
                'acoustic_features': acoustic_features,
                'spectrographic_features': spectrographic_features,
                'grbas_results': grbas_results,
                'clinical_results': clinical_results,
                'reports': reports,
                'visualizations': visualizations,
                'quality_assessment': self._assess_analysis_quality(
                    processing_info, acoustic_features, grbas_results
                )
            }
            
            # Store for potential follow-up analysis
            self.last_analysis = results
            
            print("✅ Voice analysis complete!")
            return results
            
        except Exception as e:
            error_msg = f"Voice analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Return error information
            return {
                'error': error_msg,
                'traceback': traceback.format_exc(),
                'metadata': {
                    'file_path': file_path,
                    'analysis_timestamp': datetime.datetime.now().isoformat(),
                    'voice_lab_version': self.version
                }
            }
    
    def analyze_audio_array(self,
                          audio: np.ndarray,
                          voice_type: Optional[str] = None,
                          patient_info: Optional[str] = None,
                          task_description: Optional[str] = None,
                          generate_reports: bool = True,
                          generate_visualizations: bool = None) -> Dict[str, Any]:
        """
        Complete voice analysis pipeline for audio array
        
        Args:
            audio: Audio signal as numpy array
            voice_type: 'male', 'female', or None for auto-detection
            patient_info: Patient identifier or description  
            task_description: Task description
            generate_reports: Whether to generate formatted reports
            generate_visualizations: Whether to generate plots
            
        Returns:
            Comprehensive analysis results dictionary
        """
        
        if generate_visualizations is None:
            generate_visualizations = self.enable_visualizations
        
        try:
            analysis_start = datetime.datetime.now()
            
            # Step 1: Audio preprocessing (skip file loading)
            print("🎵 Preprocessing audio...")
            processed_audio, processing_info = self.audio_processor.preprocess_audio(
                audio, self.sr
            )
            
            # Add metadata
            full_processing_info = {
                'file_metadata': {
                    'source': 'audio_array',
                    'original_sr': self.sr,
                    'duration': len(audio) / self.sr,
                    'channels': 1,
                    'format': 'array'
                },
                'processing_info': processing_info
            }
            
            # Continue with same pipeline as file analysis
            print("🔬 Extracting acoustic features...")
            acoustic_features = self.acoustic_analyzer.extract_all_features(
                processed_audio, voice_type=voice_type
            )
            
            print("📊 Analyzing spectrogram...")
            spectrographic_features = self.spectrographic_analyzer.analyze_spectrogram(processed_audio)
            
            print("👂 Estimating perceptual ratings...")
            grbas_results = self.grbas_estimator.estimate_grbas(
                acoustic_features, spectrographic_features
            )
            
            print("🏥 Mapping clinical scenarios...")
            clinical_results = self.clinical_mapper.analyze_clinical_scenarios(
                acoustic_features, grbas_results, spectrographic_features
            )
            
            reports = None
            if generate_reports:
                print("📋 Generating reports...")
                reports = self.report_generator.generate_complete_report(
                    acoustic_features, grbas_results, spectrographic_features,
                    clinical_results, patient_info, include_plots=generate_visualizations
                )
            
            visualizations = None
            if generate_visualizations and self.visualizer:
                print("📈 Creating visualizations...")
                visualizations = self.visualizer.create_comprehensive_visualization(
                    processed_audio, self.sr, acoustic_features, grbas_results,
                    spectrographic_features, clinical_results
                )
            
            analysis_end = datetime.datetime.now()
            analysis_duration = (analysis_end - analysis_start).total_seconds()
            
            results = {
                'metadata': {
                    'source': 'audio_array',
                    'patient_info': patient_info or 'Audio Array',
                    'task_description': task_description or 'Voice analysis',
                    'voice_type': voice_type,
                    'analysis_timestamp': analysis_end.isoformat(),
                    'analysis_duration_seconds': analysis_duration,
                    'voice_lab_version': self.version,
                    'audio_duration_seconds': len(processed_audio) / self.sr
                },
                'audio_processing': full_processing_info,
                'acoustic_features': acoustic_features,
                'spectrographic_features': spectrographic_features,
                'grbas_results': grbas_results,
                'clinical_results': clinical_results,
                'reports': reports,
                'visualizations': visualizations,
                'quality_assessment': self._assess_analysis_quality(
                    full_processing_info, acoustic_features, grbas_results
                )
            }
            
            self.last_analysis = results
            print("✅ Voice analysis complete!")
            return results
            
        except Exception as e:
            error_msg = f"Voice analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                'error': error_msg,
                'traceback': traceback.format_exc(),
                'metadata': {
                    'source': 'audio_array',
                    'analysis_timestamp': datetime.datetime.now().isoformat(),
                    'voice_lab_version': self.version
                }
            }
    
    def get_clinical_summary(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Get concise clinical summary from analysis results
        
        Args:
            results: Analysis results (uses last analysis if None)
            
        Returns:
            Formatted clinical summary string
        """
        
        if results is None:
            results = self.last_analysis
            
        if results is None or 'error' in results:
            return "No valid analysis results available."
        
        if results.get('reports', {}).get('clinical_summary'):
            return results['reports']['clinical_summary']
        else:
            return self._generate_quick_summary(results)
    
    def save_reports(self, 
                    output_dir: str,
                    results: Optional[Dict[str, Any]] = None,
                    formats: List[str] = None) -> Dict[str, str]:
        """
        Save analysis reports to files
        
        Args:
            output_dir: Output directory for reports
            results: Analysis results (uses last analysis if None)
            formats: List of formats to save ('json', 'html', 'txt', 'clinical_summary')
            
        Returns:
            Dictionary mapping format to saved file path
        """
        
        if results is None:
            results = self.last_analysis
            
        if results is None or 'error' in results:
            raise ValueError("No valid analysis results available to save.")
        
        if formats is None:
            formats = ['json', 'html', 'clinical_summary']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_info = results['metadata'].get('patient_info', 'voice_analysis')
        
        # Clean patient info for filename
        clean_patient = "".join(c for c in patient_info if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_patient = clean_patient.replace(' ', '_')
        
        base_filename = f"{clean_patient}_{timestamp}"
        
        saved_files = {}
        reports = results.get('reports', {})
        
        # Save each requested format
        for format_name in formats:
            try:
                if format_name == 'json' and 'json' in reports:
                    filepath = os.path.join(output_dir, f"{base_filename}_report.json")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(reports['json'])
                    saved_files['json'] = filepath
                    
                elif format_name == 'html' and 'html' in reports:
                    filepath = os.path.join(output_dir, f"{base_filename}_report.html")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(reports['html'])
                    saved_files['html'] = filepath
                    
                elif format_name == 'txt' and 'detailed_report' in reports:
                    filepath = os.path.join(output_dir, f"{base_filename}_detailed.txt")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(reports['detailed_report'])
                    saved_files['txt'] = filepath
                    
                elif format_name == 'clinical_summary' and 'clinical_summary' in reports:
                    filepath = os.path.join(output_dir, f"{base_filename}_summary.txt")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(reports['clinical_summary'])
                    saved_files['clinical_summary'] = filepath
                    
            except Exception as e:
                warnings.warn(f"Failed to save {format_name} report: {str(e)}")
        
        return saved_files
    
    def get_grbas_string(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Get GRBAS rating string (e.g., 'G1R2B2A1S0')
        
        Args:
            results: Analysis results (uses last analysis if None)
            
        Returns:
            GRBAS string
        """
        
        if results is None:
            results = self.last_analysis
            
        if results is None or 'error' in results:
            return "G?R?B?A?S?"
        
        grbas = results.get('grbas_results', {})
        return f"G{grbas.get('G', '?')}R{grbas.get('R', '?')}B{grbas.get('B', '?')}A{grbas.get('A', '?')}S{grbas.get('S', '?')}"
    
    def get_primary_impression(self, results: Optional[Dict[str, Any]] = None) -> str:
        """
        Get primary clinical impression
        
        Args:
            results: Analysis results (uses last analysis if None)
            
        Returns:
            Primary impression string
        """
        
        if results is None:
            results = self.last_analysis
            
        if results is None or 'error' in results:
            return "No analysis available"
        
        clinical = results.get('clinical_results', {})
        primary = clinical.get('primary_impression', {})
        
        condition = primary.get('condition', 'Unknown').replace('_', ' ').title()
        confidence = primary.get('confidence', 0)
        
        return f"{condition} (confidence: {confidence:.0%})"
    
    def get_red_flags(self, results: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get list of red flag findings
        
        Args:
            results: Analysis results (uses last analysis if None)
            
        Returns:
            List of red flag strings
        """
        
        if results is None:
            results = self.last_analysis
            
        if results is None or 'error' in results:
            return []
        
        clinical = results.get('clinical_results', {})
        return clinical.get('red_flags', [])
    
    def _assess_analysis_quality(self, 
                               processing_info: Dict[str, Any],
                               acoustic_features: Dict[str, float],
                               grbas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of analysis"""
        
        quality_assessment = {
            'overall_quality': 'good',
            'reliability_score': 0.8,
            'warnings': [],
            'limitations': []
        }
        
        try:
            # Audio quality checks
            audio_quality = processing_info['processing_info'].get('quality_metrics', {})
            
            if not audio_quality.get('is_good_quality', True):
                quality_assessment['warnings'].append('Poor audio quality detected')
                quality_assessment['reliability_score'] *= 0.8
            
            if audio_quality.get('duration_seconds', 0) < 2:
                quality_assessment['warnings'].append('Very short audio duration')
                quality_assessment['reliability_score'] *= 0.7
            
            if audio_quality.get('clipping_percentage', 0) > 1:
                quality_assessment['warnings'].append('Audio clipping detected')
                quality_assessment['reliability_score'] *= 0.9
            
            if audio_quality.get('estimated_snr_db', 20) < 15:
                quality_assessment['warnings'].append('Low signal-to-noise ratio')
                quality_assessment['reliability_score'] *= 0.8
            
            # Feature extraction quality
            voiced_fraction = acoustic_features.get('voiced_fraction', 1.0)
            if voiced_fraction < 0.5:
                quality_assessment['warnings'].append('Low voiced fraction - limited voicing detected')
                quality_assessment['reliability_score'] *= 0.7
            
            # GRBAS confidence
            grbas_confidence = grbas_results.get('confidence', {}).get('overall', 1.0)
            if grbas_confidence < 0.6:
                quality_assessment['warnings'].append('Low confidence in perceptual ratings')
                quality_assessment['reliability_score'] *= 0.8
            
            # Set overall quality level
            if quality_assessment['reliability_score'] >= 0.8:
                quality_assessment['overall_quality'] = 'excellent'
            elif quality_assessment['reliability_score'] >= 0.6:
                quality_assessment['overall_quality'] = 'good'
            elif quality_assessment['reliability_score'] >= 0.4:
                quality_assessment['overall_quality'] = 'fair'
            else:
                quality_assessment['overall_quality'] = 'poor'
            
        except Exception as e:
            quality_assessment['warnings'].append(f'Quality assessment error: {str(e)}')
            quality_assessment['overall_quality'] = 'uncertain'
            quality_assessment['reliability_score'] = 0.5
        
        return quality_assessment
    
    def _generate_quick_summary(self, results: Dict[str, Any]) -> str:
        """Generate quick summary if formal report is not available"""
        
        try:
            grbas = results.get('grbas_results', {})
            primary = results.get('clinical_results', {}).get('primary_impression', {})
            
            grbas_str = self.get_grbas_string(results)
            impression = self.get_primary_impression(results)
            
            summary = f"""Voice Lab GPT Quick Summary
            
GRBAS: {grbas_str}
Primary Impression: {impression}

Key Findings:
• F0: {results.get('acoustic_features', {}).get('f0_mean', 0):.1f} Hz
• HNR: {results.get('acoustic_features', {}).get('hnr_mean', 0):.1f} dB
• Jitter: {results.get('acoustic_features', {}).get('jitter_local', 0)*1000:.1f}‰

Overall Quality: {results.get('quality_assessment', {}).get('overall_quality', 'Unknown')}"""
            
            return summary
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def compare_analyses(self, 
                        results1: Dict[str, Any], 
                        results2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two voice analysis results
        
        Args:
            results1: First analysis results
            results2: Second analysis results
            
        Returns:
            Comparison results dictionary
        """
        
        if 'error' in results1 or 'error' in results2:
            return {'error': 'Cannot compare analyses with errors'}
        
        comparison = {
            'timestamp': datetime.datetime.now().isoformat(),
            'grbas_comparison': {},
            'acoustic_comparison': {},
            'clinical_comparison': {},
            'summary': ''
        }
        
        try:
            # GRBAS comparison
            grbas1 = results1.get('grbas_results', {})
            grbas2 = results2.get('grbas_results', {})
            
            for dim in ['G', 'R', 'B', 'A', 'S']:
                val1 = grbas1.get(dim, 0)
                val2 = grbas2.get(dim, 0)
                comparison['grbas_comparison'][dim] = {
                    'before': val1,
                    'after': val2,
                    'change': val2 - val1
                }
            
            # Key acoustic measures comparison
            acoustic1 = results1.get('acoustic_features', {})
            acoustic2 = results2.get('acoustic_features', {})
            
            key_measures = ['f0_mean', 'jitter_local', 'shimmer_local', 'hnr_mean', 'cpp_mean']
            
            for measure in key_measures:
                val1 = acoustic1.get(measure, 0)
                val2 = acoustic2.get(measure, 0)
                change_pct = ((val2 - val1) / (val1 + 1e-10)) * 100 if val1 != 0 else 0
                
                comparison['acoustic_comparison'][measure] = {
                    'before': val1,
                    'after': val2,
                    'change_absolute': val2 - val1,
                    'change_percent': change_pct
                }
            
            # Clinical impression comparison
            clinical1 = results1.get('clinical_results', {})
            clinical2 = results2.get('clinical_results', {})
            
            impression1 = clinical1.get('primary_impression', {}).get('condition', 'Unknown')
            impression2 = clinical2.get('primary_impression', {}).get('condition', 'Unknown')
            
            comparison['clinical_comparison'] = {
                'impression_before': impression1.replace('_', ' ').title(),
                'impression_after': impression2.replace('_', ' ').title(),
                'impression_changed': impression1 != impression2
            }
            
            # Generate summary
            grbas_changes = [abs(comparison['grbas_comparison'][dim]['change']) 
                           for dim in ['G', 'R', 'B', 'A', 'S']]
            avg_grbas_change = np.mean(grbas_changes)
            
            if avg_grbas_change < 0.5:
                change_level = "minimal"
            elif avg_grbas_change < 1.0:
                change_level = "moderate"
            else:
                change_level = "significant"
            
            comparison['summary'] = f"Comparison shows {change_level} changes in voice quality parameters."
            
        except Exception as e:
            comparison['error'] = f"Comparison failed: {str(e)}"
        
        return comparison