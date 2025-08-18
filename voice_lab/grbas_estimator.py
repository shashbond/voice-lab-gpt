"""
GRBAS perceptual estimation for Voice Lab GPT
Uses rule-based approach to estimate perceptual voice quality ratings
"""

import numpy as np
from typing import Dict, Tuple
import warnings

class GRBASEstimator:
    """Estimates GRBAS perceptual ratings from acoustic features"""
    
    def __init__(self):
        """Initialize GRBAS estimator with rule thresholds"""
        
        # Define thresholds for each GRBAS dimension
        self.thresholds = {
            'G': {  # Grade (overall severity)
                'excellent': {'hnr': 15, 'jitter': 0.005, 'shimmer': 0.03, 'cpp': 15},
                'good': {'hnr': 10, 'jitter': 0.01, 'shimmer': 0.05, 'cpp': 10},
                'fair': {'hnr': 5, 'jitter': 0.02, 'shimmer': 0.08, 'cpp': 5},
                'poor': {'hnr': 0, 'jitter': 0.05, 'shimmer': 0.15, 'cpp': 0}
            },
            'R': {  # Roughness
                'jitter_severe': 0.04,
                'jitter_moderate': 0.025,
                'jitter_mild': 0.015,
                'shimmer_severe': 0.12,
                'shimmer_moderate': 0.08,
                'shimmer_mild': 0.05,
                'aperiodicity_severe': 0.7,
                'aperiodicity_moderate': 0.4,
                'aperiodicity_mild': 0.2
            },
            'B': {  # Breathiness
                'hnr_severe': 5,
                'hnr_moderate': 10,
                'hnr_mild': 15,
                'cpp_severe': 5,
                'cpp_moderate': 10,
                'cpp_mild': 15,
                'spectral_tilt_severe': -15,
                'spectral_tilt_moderate': -10,
                'spectral_tilt_mild': -5,
                'breath_noise_severe': -20,
                'breath_noise_moderate': -30,
                'breath_noise_mild': -40
            },
            'A': {  # Asthenia (weakness)
                'intensity_low': 50,
                'intensity_moderate': 60,
                'intensity_mild': 70,
                'f0_range_low': 2,
                'f0_range_moderate': 5,
                'f0_range_mild': 10,
                'dynamic_range_low': 10,
                'dynamic_range_moderate': 20,
                'dynamic_range_mild': 30
            },
            'S': {  # Strain
                'hf_energy_high': 0.3,
                'hf_energy_moderate': 0.2,
                'hf_energy_mild': 0.1,
                'formant_constricted': 'poor',
                'spectral_tilt_positive': 0,
                'tension_indicators': 0.5
            }
        }
    
    def estimate_grbas(self, 
                      acoustic_features: Dict[str, float],
                      spectrographic_features: Dict[str, any]) -> Dict[str, any]:
        """
        Estimate GRBAS ratings from features
        
        Args:
            acoustic_features: Dictionary of acoustic measures
            spectrographic_features: Dictionary of spectrographic findings
            
        Returns:
            Dictionary with GRBAS ratings and confidence scores
        """
        
        # Extract and validate features
        features = self._extract_relevant_features(acoustic_features, spectrographic_features)
        
        # Estimate each GRBAS dimension
        G_rating, G_confidence = self._estimate_grade(features)
        R_rating, R_confidence = self._estimate_roughness(features)
        B_rating, B_confidence = self._estimate_breathiness(features)
        A_rating, A_confidence = self._estimate_asthenia(features)
        S_rating, S_confidence = self._estimate_strain(features)
        
        # Overall confidence based on feature availability and quality
        overall_confidence = self._calculate_overall_confidence(
            features, [G_confidence, R_confidence, B_confidence, A_confidence, S_confidence]
        )
        
        # Create reasoning explanations
        reasoning = self._generate_reasoning(features, {
            'G': G_rating, 'R': R_rating, 'B': B_rating, 'A': A_rating, 'S': S_rating
        })
        
        return {
            'G': G_rating,
            'R': R_rating,
            'B': B_rating,
            'A': A_rating,
            'S': S_rating,
            'confidence': {
                'G': G_confidence,
                'R': R_confidence,
                'B': B_confidence,
                'A': A_confidence,
                'S': S_confidence,
                'overall': overall_confidence
            },
            'reasoning': reasoning,
            'feature_quality': self._assess_feature_quality(features)
        }
    
    def _extract_relevant_features(self, 
                                 acoustic_features: Dict[str, float], 
                                 spectrographic_features: Dict[str, any]) -> Dict[str, float]:
        """Extract and combine relevant features for GRBAS estimation"""
        
        features = {}
        
        # Acoustic features
        features['hnr_mean'] = acoustic_features.get('hnr_mean', 0)
        features['jitter_local'] = acoustic_features.get('jitter_local', 0)
        features['shimmer_local'] = acoustic_features.get('shimmer_local', 0)
        features['cpp_mean'] = acoustic_features.get('cpp_mean', 0)
        features['spectral_tilt'] = acoustic_features.get('spectral_tilt', 0)
        features['f0_std'] = acoustic_features.get('f0_std', 0)
        features['intensity_mean'] = acoustic_features.get('intensity_mean', 0)
        features['high_freq_ratio'] = acoustic_features.get('high_freq_ratio', 0)
        features['dynamic_range_db'] = acoustic_features.get('dynamic_range_db', 0)
        features['voiced_fraction'] = acoustic_features.get('voiced_fraction', 1.0)
        
        # Spectrographic features
        features['aperiodicity_score'] = spectrographic_features.get('aperiodicity_score', 0)
        features['breath_noise_level'] = spectrographic_features.get('breath_noise_level', -60)
        features['harmonic_clarity_score'] = spectrographic_features.get('harmonic_clarity_score', 0)
        features['subharmonic_strength'] = spectrographic_features.get('subharmonic_strength', 0)
        features['very_hf_energy_ratio'] = spectrographic_features.get('very_hf_energy_ratio', 0)
        features['formant_clarity'] = spectrographic_features.get('formant_clarity', 'unknown')
        features['voice_breaks_percentage'] = spectrographic_features.get('voice_breaks_percentage', 0)
        
        return features
    
    def _estimate_grade(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Estimate Grade (G) - overall voice quality"""
        
        # Multiple indicators contribute to grade
        indicators = []
        
        # HNR contribution
        hnr = features['hnr_mean']
        if hnr >= 15:
            hnr_score = 0
        elif hnr >= 10:
            hnr_score = 1
        elif hnr >= 5:
            hnr_score = 2
        else:
            hnr_score = 3
        indicators.append(('hnr', hnr_score, 0.3))
        
        # Jitter contribution
        jitter = features['jitter_local']
        if jitter <= 0.005:
            jitter_score = 0
        elif jitter <= 0.01:
            jitter_score = 1
        elif jitter <= 0.02:
            jitter_score = 2
        else:
            jitter_score = 3
        indicators.append(('jitter', jitter_score, 0.25))
        
        # Shimmer contribution
        shimmer = features['shimmer_local']
        if shimmer <= 0.03:
            shimmer_score = 0
        elif shimmer <= 0.05:
            shimmer_score = 1
        elif shimmer <= 0.08:
            shimmer_score = 2
        else:
            shimmer_score = 3
        indicators.append(('shimmer', shimmer_score, 0.25))
        
        # CPP contribution
        cpp = features['cpp_mean']
        if cpp >= 15:
            cpp_score = 0
        elif cpp >= 10:
            cpp_score = 1
        elif cpp >= 5:
            cpp_score = 2
        else:
            cpp_score = 3
        indicators.append(('cpp', cpp_score, 0.2))
        
        # Calculate weighted average
        weighted_score = sum(score * weight for _, score, weight in indicators)
        grade = int(round(weighted_score))
        grade = max(0, min(3, grade))  # Clamp to 0-3 range
        
        # Calculate confidence based on consistency of indicators
        scores = [score for _, score, _ in indicators]
        consistency = 1 - (np.std(scores) / 3.0)  # Normalized standard deviation
        confidence = max(0.5, min(1.0, consistency * 0.8 + 0.2))
        
        return grade, confidence
    
    def _estimate_roughness(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Estimate Roughness (R)"""
        
        indicators = []
        
        # Jitter indicators
        jitter = features['jitter_local']
        if jitter >= self.thresholds['R']['jitter_severe']:
            jitter_score = 3
        elif jitter >= self.thresholds['R']['jitter_moderate']:
            jitter_score = 2
        elif jitter >= self.thresholds['R']['jitter_mild']:
            jitter_score = 1
        else:
            jitter_score = 0
        indicators.append(('jitter', jitter_score, 0.4))
        
        # Shimmer indicators
        shimmer = features['shimmer_local']
        if shimmer >= self.thresholds['R']['shimmer_severe']:
            shimmer_score = 3
        elif shimmer >= self.thresholds['R']['shimmer_moderate']:
            shimmer_score = 2
        elif shimmer >= self.thresholds['R']['shimmer_mild']:
            shimmer_score = 1
        else:
            shimmer_score = 0
        indicators.append(('shimmer', shimmer_score, 0.3))
        
        # Aperiodicity indicators
        aperiodicity = features['aperiodicity_score']
        if aperiodicity >= self.thresholds['R']['aperiodicity_severe']:
            aperiodicity_score = 3
        elif aperiodicity >= self.thresholds['R']['aperiodicity_moderate']:
            aperiodicity_score = 2
        elif aperiodicity >= self.thresholds['R']['aperiodicity_mild']:
            aperiodicity_score = 1
        else:
            aperiodicity_score = 0
        indicators.append(('aperiodicity', aperiodicity_score, 0.2))
        
        # Subharmonics
        subharmonic_strength = features['subharmonic_strength']
        if subharmonic_strength > 0.3:
            subharmonic_score = 2
        elif subharmonic_strength > 0.1:
            subharmonic_score = 1
        else:
            subharmonic_score = 0
        indicators.append(('subharmonic', subharmonic_score, 0.1))
        
        # Calculate weighted score
        weighted_score = sum(score * weight for _, score, weight in indicators)
        roughness = int(round(weighted_score))
        roughness = max(0, min(3, roughness))
        
        # Confidence calculation
        scores = [score for _, score, _ in indicators]
        if len(scores) > 0:
            consistency = 1 - (np.std(scores) / 3.0)
            confidence = max(0.6, min(1.0, consistency * 0.7 + 0.3))
        else:
            confidence = 0.5
        
        return roughness, confidence
    
    def _estimate_breathiness(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Estimate Breathiness (B)"""
        
        indicators = []
        
        # HNR (inverse relationship with breathiness)
        hnr = features['hnr_mean']
        if hnr <= self.thresholds['B']['hnr_severe']:
            hnr_score = 3
        elif hnr <= self.thresholds['B']['hnr_moderate']:
            hnr_score = 2
        elif hnr <= self.thresholds['B']['hnr_mild']:
            hnr_score = 1
        else:
            hnr_score = 0
        indicators.append(('hnr', hnr_score, 0.35))
        
        # CPP (inverse relationship)
        cpp = features['cpp_mean']
        if cpp <= self.thresholds['B']['cpp_severe']:
            cpp_score = 3
        elif cpp <= self.thresholds['B']['cpp_moderate']:
            cpp_score = 2
        elif cpp <= self.thresholds['B']['cpp_mild']:
            cpp_score = 1
        else:
            cpp_score = 0
        indicators.append(('cpp', cpp_score, 0.25))
        
        # Spectral tilt (more negative = more breathiness)
        spectral_tilt = features['spectral_tilt']
        if spectral_tilt <= self.thresholds['B']['spectral_tilt_severe']:
            tilt_score = 3
        elif spectral_tilt <= self.thresholds['B']['spectral_tilt_moderate']:
            tilt_score = 2
        elif spectral_tilt <= self.thresholds['B']['spectral_tilt_mild']:
            tilt_score = 1
        else:
            tilt_score = 0
        indicators.append(('spectral_tilt', tilt_score, 0.2))
        
        # Breath noise level
        breath_noise = features['breath_noise_level']
        if breath_noise >= self.thresholds['B']['breath_noise_severe']:
            breath_score = 3
        elif breath_noise >= self.thresholds['B']['breath_noise_moderate']:
            breath_score = 2
        elif breath_noise >= self.thresholds['B']['breath_noise_mild']:
            breath_score = 1
        else:
            breath_score = 0
        indicators.append(('breath_noise', breath_score, 0.2))
        
        # Calculate weighted score
        weighted_score = sum(score * weight for _, score, weight in indicators)
        breathiness = int(round(weighted_score))
        breathiness = max(0, min(3, breathiness))
        
        # Confidence calculation
        scores = [score for _, score, _ in indicators]
        consistency = 1 - (np.std(scores) / 3.0)
        confidence = max(0.6, min(1.0, consistency * 0.8 + 0.2))
        
        return breathiness, confidence
    
    def _estimate_asthenia(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Estimate Asthenia (A) - weakness"""
        
        indicators = []
        
        # Intensity level
        intensity = features['intensity_mean']
        if intensity <= self.thresholds['A']['intensity_low']:
            intensity_score = 3
        elif intensity <= self.thresholds['A']['intensity_moderate']:
            intensity_score = 2
        elif intensity <= self.thresholds['A']['intensity_mild']:
            intensity_score = 1
        else:
            intensity_score = 0
        indicators.append(('intensity', intensity_score, 0.4))
        
        # F0 variability (reduced in asthenia)
        f0_std = features['f0_std']
        if f0_std <= self.thresholds['A']['f0_range_low']:
            f0_score = 3
        elif f0_std <= self.thresholds['A']['f0_range_moderate']:
            f0_score = 2
        elif f0_std <= self.thresholds['A']['f0_range_mild']:
            f0_score = 1
        else:
            f0_score = 0
        indicators.append(('f0_variability', f0_score, 0.3))
        
        # Dynamic range
        dynamic_range = features['dynamic_range_db']
        if dynamic_range <= self.thresholds['A']['dynamic_range_low']:
            dynamic_score = 3
        elif dynamic_range <= self.thresholds['A']['dynamic_range_moderate']:
            dynamic_score = 2
        elif dynamic_range <= self.thresholds['A']['dynamic_range_mild']:
            dynamic_score = 1
        else:
            dynamic_score = 0
        indicators.append(('dynamic_range', dynamic_score, 0.3))
        
        # Calculate weighted score
        weighted_score = sum(score * weight for _, score, weight in indicators)
        asthenia = int(round(weighted_score))
        asthenia = max(0, min(3, asthenia))
        
        # Confidence calculation
        scores = [score for _, score, _ in indicators]
        consistency = 1 - (np.std(scores) / 3.0)
        confidence = max(0.5, min(1.0, consistency * 0.7 + 0.3))
        
        return asthenia, confidence
    
    def _estimate_strain(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Estimate Strain (S)"""
        
        indicators = []
        
        # High-frequency energy (increased in strain)
        hf_energy = features['very_hf_energy_ratio']
        if hf_energy >= self.thresholds['S']['hf_energy_high']:
            hf_score = 3
        elif hf_energy >= self.thresholds['S']['hf_energy_moderate']:
            hf_score = 2
        elif hf_energy >= self.thresholds['S']['hf_energy_mild']:
            hf_score = 1
        else:
            hf_score = 0
        indicators.append(('hf_energy', hf_score, 0.4))
        
        # Spectral tilt (less negative or positive in strain)
        spectral_tilt = features['spectral_tilt']
        if spectral_tilt >= self.thresholds['S']['spectral_tilt_positive']:
            tilt_score = 2
        elif spectral_tilt >= -5:
            tilt_score = 1
        else:
            tilt_score = 0
        indicators.append(('spectral_tilt', tilt_score, 0.3))
        
        # Formant clarity (may be reduced due to constriction)
        formant_clarity = features['formant_clarity']
        if formant_clarity == 'poor':
            formant_score = 2
        elif formant_clarity == 'fair':
            formant_score = 1
        else:
            formant_score = 0
        indicators.append(('formant_clarity', formant_score, 0.2))
        
        # Voice breaks (may increase with strain)
        voice_breaks = features['voice_breaks_percentage']
        if voice_breaks > 10:
            breaks_score = 2
        elif voice_breaks > 5:
            breaks_score = 1
        else:
            breaks_score = 0
        indicators.append(('voice_breaks', breaks_score, 0.1))
        
        # Calculate weighted score
        weighted_score = sum(score * weight for _, score, weight in indicators)
        strain = int(round(weighted_score))
        strain = max(0, min(3, strain))
        
        # Confidence calculation
        scores = [score for _, score, _ in indicators]
        consistency = 1 - (np.std(scores) / 3.0)
        confidence = max(0.4, min(1.0, consistency * 0.6 + 0.4))
        
        return strain, confidence
    
    def _calculate_overall_confidence(self, 
                                    features: Dict[str, float], 
                                    individual_confidences: list) -> float:
        """Calculate overall confidence score"""
        
        # Base confidence on individual dimension confidences
        mean_confidence = np.mean(individual_confidences)
        
        # Adjust for feature quality and availability
        feature_availability = self._assess_feature_availability(features)
        signal_quality = self._assess_signal_quality(features)
        
        # Combine factors
        overall_confidence = mean_confidence * feature_availability * signal_quality
        
        return float(max(0.3, min(1.0, overall_confidence)))
    
    def _assess_feature_availability(self, features: Dict[str, float]) -> float:
        """Assess how many key features are available and valid"""
        
        key_features = ['hnr_mean', 'jitter_local', 'shimmer_local', 'cpp_mean', 
                       'spectral_tilt', 'intensity_mean']
        
        available_count = 0
        for feature in key_features:
            if feature in features and features[feature] != 0:
                available_count += 1
        
        availability_ratio = available_count / len(key_features)
        return max(0.5, availability_ratio)
    
    def _assess_signal_quality(self, features: Dict[str, float]) -> float:
        """Assess overall signal quality for analysis"""
        
        quality_score = 1.0
        
        # Voiced fraction (should be reasonable for voice analysis)
        voiced_fraction = features.get('voiced_fraction', 1.0)
        if voiced_fraction < 0.3:
            quality_score *= 0.5
        elif voiced_fraction < 0.5:
            quality_score *= 0.7
        
        # Harmonic clarity
        harmonic_clarity = features.get('harmonic_clarity_score', 5)
        if harmonic_clarity < 2:
            quality_score *= 0.6
        elif harmonic_clarity < 5:
            quality_score *= 0.8
        
        return quality_score
    
    def _assess_feature_quality(self, features: Dict[str, float]) -> Dict[str, str]:
        """Assess quality of extracted features"""
        
        quality_assessment = {}
        
        # HNR quality
        if features['hnr_mean'] == 0:
            quality_assessment['hnr'] = 'unavailable'
        elif features['voiced_fraction'] < 0.3:
            quality_assessment['hnr'] = 'low_reliability'
        else:
            quality_assessment['hnr'] = 'good'
        
        # Jitter/shimmer quality
        if features['jitter_local'] == 0 or features['shimmer_local'] == 0:
            quality_assessment['periodicity'] = 'unavailable'
        elif features['voiced_fraction'] < 0.5:
            quality_assessment['periodicity'] = 'low_reliability'
        else:
            quality_assessment['periodicity'] = 'good'
        
        # Spectral quality
        if features['cpp_mean'] == 0:
            quality_assessment['spectral'] = 'unavailable'
        else:
            quality_assessment['spectral'] = 'good'
        
        return quality_assessment
    
    def _generate_reasoning(self, 
                          features: Dict[str, float], 
                          ratings: Dict[str, int]) -> Dict[str, str]:
        """Generate reasoning for each GRBAS rating"""
        
        reasoning = {}
        
        # Grade reasoning
        if ratings['G'] == 0:
            reasoning['G'] = "Normal voice quality with good acoustic measures"
        elif ratings['G'] == 1:
            reasoning['G'] = "Mild voice quality deviation with some acoustic irregularities"
        elif ratings['G'] == 2:
            reasoning['G'] = "Moderate voice quality impairment with notable acoustic abnormalities"
        else:
            reasoning['G'] = "Severe voice quality impairment with significant acoustic deviations"
        
        # Roughness reasoning
        jitter = features['jitter_local']
        shimmer = features['shimmer_local']
        if ratings['R'] == 0:
            reasoning['R'] = "No roughness detected"
        elif ratings['R'] == 1:
            reasoning['R'] = f"Mild roughness (jitter: {jitter:.3f}, shimmer: {shimmer:.3f})"
        elif ratings['R'] == 2:
            reasoning['R'] = f"Moderate roughness with elevated perturbation measures"
        else:
            reasoning['R'] = f"Severe roughness with high irregularity in vocal fold vibration"
        
        # Breathiness reasoning
        hnr = features['hnr_mean']
        if ratings['B'] == 0:
            reasoning['B'] = "No breathiness detected"
        elif ratings['B'] == 1:
            reasoning['B'] = f"Mild breathiness (HNR: {hnr:.1f} dB)"
        elif ratings['B'] == 2:
            reasoning['B'] = f"Moderate breathiness with reduced harmonic-to-noise ratio"
        else:
            reasoning['B'] = f"Severe breathiness with significant noise component"
        
        # Asthenia reasoning
        intensity = features['intensity_mean']
        if ratings['A'] == 0:
            reasoning['A'] = "Normal vocal strength"
        elif ratings['A'] == 1:
            reasoning['A'] = f"Mild weakness (intensity: {intensity:.1f} dB)"
        elif ratings['A'] == 2:
            reasoning['A'] = f"Moderate weakness with reduced vocal projection"
        else:
            reasoning['A'] = f"Severe weakness with very low vocal intensity"
        
        # Strain reasoning
        hf_ratio = features['very_hf_energy_ratio']
        if ratings['S'] == 0:
            reasoning['S'] = "No strain detected"
        elif ratings['S'] == 1:
            reasoning['S'] = f"Mild strain with some high-frequency emphasis"
        elif ratings['S'] == 2:
            reasoning['S'] = f"Moderate strain with notable constriction"
        else:
            reasoning['S'] = f"Severe strain with significant vocal constriction"
        
        return reasoning