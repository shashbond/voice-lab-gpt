"""
Test suite for Voice Lab GPT
Comprehensive testing of all system components
"""

import unittest
import numpy as np
import tempfile
import os
import sys
import warnings
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import Voice Lab GPT components
from voice_lab import VoiceLabGPT
from voice_lab.audio_processor import AudioProcessor
from voice_lab.acoustic_analyzer import AcousticAnalyzer
from voice_lab.spectrographic_analyzer import SpectrographicAnalyzer
from voice_lab.grbas_estimator import GRBASEstimator
from voice_lab.clinical_mapper import ClinicalMapper
from voice_lab.report_generator import ReportGenerator

class TestAudioProcessor(unittest.TestCase):
    """Test audio processing functionality"""
    
    def setUp(self):
        self.processor = AudioProcessor()
        
    def test_initialization(self):
        """Test AudioProcessor initialization"""
        self.assertEqual(self.processor.target_sr, 16000)
        self.assertEqual(self.processor.target_lufs, -23.0)
        self.assertIsNotNone(self.processor.meter)
    
    def test_convert_to_mono(self):
        """Test mono conversion"""
        # Test stereo to mono
        stereo_audio = np.random.randn(2, 1000)
        mono_audio = self.processor._convert_to_mono(stereo_audio)
        self.assertEqual(mono_audio.shape, (1000,))
        
        # Test already mono
        mono_input = np.random.randn(1000)
        mono_output = self.processor._convert_to_mono(mono_input)
        np.testing.assert_array_equal(mono_input, mono_output)
    
    def test_ensure_16bit_range(self):
        """Test 16-bit range normalization"""
        # Test clipping prevention
        loud_audio = np.random.randn(1000) * 10
        normalized_audio = self.processor._ensure_16bit_range(loud_audio)
        self.assertLessEqual(np.max(np.abs(normalized_audio)), 1.0)
    
    def test_assess_quality(self):
        """Test audio quality assessment"""
        # Create test audio
        sr = 16000
        duration = 2.0
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        quality_metrics = self.processor._assess_quality(audio, sr)
        
        self.assertIn('estimated_snr_db', quality_metrics)
        self.assertIn('clipping_percentage', quality_metrics)
        self.assertIn('duration_seconds', quality_metrics)
        self.assertIn('is_good_quality', quality_metrics)
    
    def test_preprocess_audio(self):
        """Test complete preprocessing pipeline"""
        # Create test audio
        sr = 22050
        audio = np.random.randn(sr * 2) * 0.5  # 2 seconds
        
        processed_audio, info = self.processor.preprocess_audio(audio, sr)
        
        self.assertIsInstance(processed_audio, np.ndarray)
        self.assertIn('steps_applied', info)
        self.assertIn('final_duration', info)

class TestAcousticAnalyzer(unittest.TestCase):
    """Test acoustic feature extraction"""
    
    def setUp(self):
        self.analyzer = AcousticAnalyzer()
    
    def test_initialization(self):
        """Test AcousticAnalyzer initialization"""
        self.assertEqual(self.analyzer.sr, 16000)
    
    def test_synthetic_audio_analysis(self):
        """Test analysis with synthetic audio"""
        # Generate synthetic vowel
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        f0 = 150  # Hz
        
        # Create harmonic series
        audio = np.zeros_like(t)
        for harmonic in range(1, 6):
            amplitude = 1.0 / harmonic
            audio += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)
        
        # Add small amount of jitter/shimmer
        jitter = 0.01 * np.random.randn(len(t))
        audio *= (1 + 0.02 * np.random.randn(len(t)))  # Shimmer
        
        # Analyze
        features = self.analyzer.extract_all_features(audio)
        
        # Check that key features are extracted
        self.assertIn('f0_mean', features)
        self.assertIn('jitter_local', features)
        self.assertIn('shimmer_local', features)
        self.assertIn('hnr_mean', features)
        self.assertIn('cpp_mean', features)
        
        # Check F0 is reasonable
        self.assertAlmostEqual(features['f0_mean'], f0, delta=20)
        
        # Check feature ranges
        self.assertGreaterEqual(features['jitter_local'], 0)
        self.assertGreaterEqual(features['shimmer_local'], 0)

class TestSpectrographicAnalyzer(unittest.TestCase):
    """Test spectrographic analysis"""
    
    def setUp(self):
        self.analyzer = SpectrographicAnalyzer()
    
    def test_initialization(self):
        """Test SpectrographicAnalyzer initialization"""
        self.assertEqual(self.analyzer.sr, 16000)
    
    def test_analyze_spectrogram(self):
        """Test spectrogram analysis"""
        # Generate test audio
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 150 * t)  # Pure tone
        
        findings = self.analyzer.analyze_spectrogram(audio)
        
        # Check required outputs
        self.assertIn('harmonic_definition', findings)
        self.assertIn('breath_noise_level', findings)
        self.assertIn('overall_aperiodicity', findings)
        self.assertIn('spectrogram_data', findings)

class TestGRBASEstimator(unittest.TestCase):
    """Test GRBAS perceptual estimation"""
    
    def setUp(self):
        self.estimator = GRBASEstimator()
    
    def test_initialization(self):
        """Test GRBASEstimator initialization"""
        self.assertIn('G', self.estimator.thresholds)
        self.assertIn('R', self.estimator.thresholds)
        self.assertIn('B', self.estimator.thresholds)
    
    def test_estimate_grbas(self):
        """Test GRBAS estimation"""
        # Create sample features
        acoustic_features = {
            'hnr_mean': 12.0,
            'jitter_local': 0.015,
            'shimmer_local': 0.06,
            'cpp_mean': 8.0,
            'spectral_tilt': -10.0,
            'f0_std': 8.0,
            'intensity_mean': 65.0,
            'high_freq_ratio': 0.15,
            'dynamic_range_db': 25.0,
            'voiced_fraction': 0.85
        }
        
        spectrographic_features = {
            'aperiodicity_score': 0.2,
            'breath_noise_level': -35,
            'harmonic_clarity_score': 6.0,
            'subharmonic_strength': 0.05,
            'very_hf_energy_ratio': 0.1,
            'formant_clarity': 'good',
            'voice_breaks_percentage': 3.0
        }
        
        results = self.estimator.estimate_grbas(acoustic_features, spectrographic_features)
        
        # Check outputs
        for dim in ['G', 'R', 'B', 'A', 'S']:
            self.assertIn(dim, results)
            self.assertIsInstance(results[dim], int)
            self.assertGreaterEqual(results[dim], 0)
            self.assertLessEqual(results[dim], 3)
        
        self.assertIn('confidence', results)
        self.assertIn('reasoning', results)

class TestClinicalMapper(unittest.TestCase):
    """Test clinical scenario mapping"""
    
    def setUp(self):
        self.mapper = ClinicalMapper()
    
    def test_initialization(self):
        """Test ClinicalMapper initialization"""
        self.assertIn('glottic_insufficiency', self.mapper.clinical_patterns)
        self.assertIn('vocal_fold_lesions', self.mapper.clinical_patterns)
    
    def test_analyze_clinical_scenarios(self):
        """Test clinical scenario analysis"""
        # Sample data representing glottic insufficiency
        acoustic_features = {
            'hnr_mean': 8.0,  # Low HNR
            'breathiness_score': 2,
            'spectral_tilt': -15.0,
            'intensity_mean': 55.0
        }
        
        grbas_results = {
            'G': 2, 'R': 1, 'B': 3, 'A': 2, 'S': 0,
            'confidence': {'overall': 0.75}
        }
        
        spectrographic_features = {
            'harmonic_definition': 'poor',
            'breath_noise_level': -25,
            'harmonic_clarity_score': 3.0
        }
        
        results = self.mapper.analyze_clinical_scenarios(
            acoustic_features, grbas_results, spectrographic_features
        )
        
        self.assertIn('primary_impression', results)
        self.assertIn('differential_diagnoses', results)
        self.assertIn('triage', results)
        self.assertIn('recommendations', results)
        self.assertIn('clinical_reasoning', results)

class TestReportGenerator(unittest.TestCase):
    """Test report generation"""
    
    def setUp(self):
        self.generator = ReportGenerator()
    
    def test_initialization(self):
        """Test ReportGenerator initialization"""
        self.assertIsInstance(self.generator.html_template, str)
        self.assertIn('<!DOCTYPE html>', self.generator.html_template)
    
    def test_generate_json_report(self):
        """Test JSON report generation"""
        sample_data = {
            'metadata': {'timestamp': '2024-01-01T00:00:00'},
            'acoustic_features': {'f0_mean': 150.0},
            'grbas_results': {'G': 1, 'R': 0, 'B': 0, 'A': 0, 'S': 0}
        }
        
        json_report = self.generator._generate_json_report(sample_data)
        self.assertIsInstance(json_report, str)
        self.assertIn('timestamp', json_report)
        self.assertIn('f0_mean', json_report)

class TestVoiceLabGPT(unittest.TestCase):
    """Test main Voice Lab GPT interface"""
    
    def setUp(self):
        self.voice_lab = VoiceLabGPT(enable_visualizations=False)
    
    def test_initialization(self):
        """Test VoiceLabGPT initialization"""
        self.assertEqual(self.voice_lab.sr, 16000)
        self.assertIsInstance(self.voice_lab.audio_processor, AudioProcessor)
        self.assertIsInstance(self.voice_lab.acoustic_analyzer, AcousticAnalyzer)
        self.assertEqual(self.voice_lab.version, "1.0.0")
    
    def test_analyze_audio_array(self):
        """Test analysis with audio array"""
        # Generate synthetic test audio
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a more realistic voice signal
        f0 = 150
        audio = np.zeros_like(t)
        
        # Add harmonics
        for h in range(1, 6):
            amplitude = 1.0 / h
            frequency = f0 * h
            audio += amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise for realism
        audio += 0.05 * np.random.randn(len(t))
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Analyze
        results = self.voice_lab.analyze_audio_array(
            audio,
            voice_type='female',
            patient_info='Test Patient',
            task_description='Sustained /a/',
            generate_reports=True,
            generate_visualizations=False
        )
        
        # Check results structure
        self.assertNotIn('error', results)
        self.assertIn('metadata', results)
        self.assertIn('acoustic_features', results)
        self.assertIn('grbas_results', results)
        self.assertIn('clinical_results', results)
        self.assertIn('quality_assessment', results)
        
        # Check GRBAS results
        grbas = results['grbas_results']
        for dim in ['G', 'R', 'B', 'A', 'S']:
            self.assertIn(dim, grbas)
            self.assertIsInstance(grbas[dim], int)
            self.assertGreaterEqual(grbas[dim], 0)
            self.assertLessEqual(grbas[dim], 3)
    
    def test_grbas_string_generation(self):
        """Test GRBAS string generation"""
        # Create mock results
        mock_results = {
            'grbas_results': {'G': 1, 'R': 2, 'B': 0, 'A': 1, 'S': 0}
        }
        
        grbas_string = self.voice_lab.get_grbas_string(mock_results)
        self.assertEqual(grbas_string, "G1R2B0A1S0")
    
    def test_primary_impression(self):
        """Test primary impression extraction"""
        mock_results = {
            'clinical_results': {
                'primary_impression': {
                    'condition': 'vocal_fold_lesions',
                    'confidence': 0.75
                }
            }
        }
        
        impression = self.voice_lab.get_primary_impression(mock_results)
        self.assertIn('Vocal Fold Lesions', impression)
        self.assertIn('75%', impression)
    
    def test_red_flags_detection(self):
        """Test red flags extraction"""
        mock_results = {
            'clinical_results': {
                'red_flags': ['Severe voice impairment', 'Very low vocal intensity']
            }
        }
        
        red_flags = self.voice_lab.get_red_flags(mock_results)
        self.assertEqual(len(red_flags), 2)
        self.assertIn('Severe voice impairment', red_flags)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_empty_audio(self):
        """Test handling of empty audio"""
        voice_lab = VoiceLabGPT(enable_visualizations=False)
        empty_audio = np.array([])
        
        results = voice_lab.analyze_audio_array(empty_audio)
        self.assertIn('error', results)
    
    def test_very_short_audio(self):
        """Test handling of very short audio"""
        voice_lab = VoiceLabGPT(enable_visualizations=False)
        short_audio = np.random.randn(1000)  # Very short
        
        with warnings.catch_warnings(record=True):
            results = voice_lab.analyze_audio_array(short_audio)
            # Should complete but with warnings
            self.assertIn('quality_assessment', results)
    
    def test_silent_audio(self):
        """Test handling of silent audio"""
        voice_lab = VoiceLabGPT(enable_visualizations=False)
        silent_audio = np.zeros(16000 * 2)  # 2 seconds of silence
        
        results = voice_lab.analyze_audio_array(silent_audio)
        # Should handle gracefully
        self.assertIsInstance(results, dict)
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file"""
        voice_lab = VoiceLabGPT(enable_visualizations=False)
        
        with self.assertRaises(FileNotFoundError):
            voice_lab.analyze_file('/nonexistent/path/to/file.wav')

class TestIntegration(unittest.TestCase):
    """Integration tests for complete system"""
    
    def test_complete_pipeline(self):
        """Test complete analysis pipeline"""
        voice_lab = VoiceLabGPT(enable_visualizations=False)
        
        # Create more realistic test signal
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Simulate different voice conditions
        conditions = [
            {'name': 'normal', 'f0': 150, 'jitter': 0.005, 'shimmer': 0.02, 'noise': 0.01},
            {'name': 'breathy', 'f0': 140, 'jitter': 0.008, 'shimmer': 0.04, 'noise': 0.08},
            {'name': 'rough', 'f0': 160, 'jitter': 0.025, 'shimmer': 0.08, 'noise': 0.02}
        ]
        
        for condition in conditions:
            # Generate audio for this condition
            f0 = condition['f0']
            audio = np.zeros_like(t)
            
            # Add harmonics with perturbations
            for h in range(1, 5):
                jitter_noise = condition['jitter'] * np.random.randn(len(t))
                shimmer_noise = condition['shimmer'] * np.random.randn(len(t))
                
                frequency = f0 * h * (1 + jitter_noise)
                amplitude = (1.0 / h) * (1 + shimmer_noise)
                
                audio += amplitude * np.sin(2 * np.pi * frequency * t)
            
            # Add noise
            audio += condition['noise'] * np.random.randn(len(t))
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Analyze
            results = voice_lab.analyze_audio_array(
                audio, 
                patient_info=f"Test - {condition['name']} voice",
                generate_reports=False,
                generate_visualizations=False
            )
            
            # Check analysis completed
            self.assertNotIn('error', results)
            
            # Basic validation
            grbas = results['grbas_results']
            self.assertIsInstance(grbas['G'], int)
            
            acoustic = results['acoustic_features']
            self.assertAlmostEqual(acoustic['f0_mean'], f0, delta=30)
            
            print(f"✅ {condition['name'].title()} voice analysis completed: "
                  f"GRBAS=G{grbas['G']}R{grbas['R']}B{grbas['B']}A{grbas['A']}S{grbas['S']}")

def run_performance_test():
    """Test analysis performance"""
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    
    import time
    
    voice_lab = VoiceLabGPT(enable_visualizations=False)
    
    # Generate test audio
    sr = 16000
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 150 * t) + 0.05 * np.random.randn(len(t))
    
    # Time the analysis
    start_time = time.time()
    results = voice_lab.analyze_audio_array(audio, generate_reports=False, generate_visualizations=False)
    end_time = time.time()
    
    analysis_time = end_time - start_time
    audio_duration = duration
    real_time_factor = analysis_time / audio_duration
    
    print(f"Audio Duration:    {audio_duration:.1f} seconds")
    print(f"Analysis Time:     {analysis_time:.2f} seconds")
    print(f"Real-time Factor:  {real_time_factor:.2f}x")
    print(f"Analysis Success:  {'✅ Yes' if 'error' not in results else '❌ No'}")
    
    if real_time_factor < 1.0:
        print("🚀 Analysis is faster than real-time!")
    else:
        print("⏱️  Analysis is slower than real-time")

if __name__ == '__main__':
    print("🧪 VOICE LAB GPT - TEST SUITE")
    print("=" * 80)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test
    try:
        run_performance_test()
    except Exception as e:
        print(f"Performance test failed: {str(e)}")
    
    print("\n" + "=" * 80)
    print("✅ TEST SUITE COMPLETED")
    print("=" * 80)