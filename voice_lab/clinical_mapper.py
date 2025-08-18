"""
Clinical scenario mapping for Voice Lab GPT
Maps acoustic/perceptual patterns to clinical impressions and recommendations
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

class ClinicalMapper:
    """Maps voice analysis results to clinical scenarios and recommendations"""
    
    def __init__(self):
        """Initialize clinical mapper with knowledge base"""
        
        # Clinical pattern definitions
        self.clinical_patterns = {
            'glottic_insufficiency': {
                'acoustic_profile': {
                    'hnr_mean': (0, 10),
                    'breathiness_score': (2, 3),
                    'spectral_tilt': (-20, -8),
                    'breath_noise_level': (-30, 0)
                },
                'perceptual_profile': {
                    'B': (2, 3),  # High breathiness
                    'A': (1, 3),  # May have weakness
                    'G': (1, 3)   # Overall impairment
                },
                'spectrographic_signs': [
                    'poor harmonic definition',
                    'elevated breath noise',
                    'reduced harmonic clarity'
                ],
                'confidence_weight': 0.8
            },
            
            'vocal_fold_lesions': {
                'acoustic_profile': {
                    'jitter_local': (0.015, 1.0),
                    'shimmer_local': (0.08, 1.0),
                    'roughness_score': (2, 3),
                    'aperiodicity_score': (0.3, 1.0)
                },
                'perceptual_profile': {
                    'R': (2, 3),  # High roughness
                    'G': (2, 3),  # Significant impairment
                    'B': (0, 2)   # Variable breathiness
                },
                'spectrographic_signs': [
                    'subharmonic presence',
                    'irregular harmonic structure',
                    'aperiodic segments'
                ],
                'confidence_weight': 0.7
            },
            
            'vocal_fold_paralysis': {
                'acoustic_profile': {
                    'hnr_mean': (0, 8),
                    'breathiness_score': (2, 3),
                    'intensity_mean': (40, 65),
                    'voiced_fraction': (0.2, 0.8)
                },
                'perceptual_profile': {
                    'B': (2, 3),  # Severe breathiness
                    'A': (2, 3),  # Weakness
                    'G': (2, 3)   # Severe impairment
                },
                'spectrographic_signs': [
                    'poor harmonic definition',
                    'extensive breath noise',
                    'voice breaks'
                ],
                'confidence_weight': 0.9
            },
            
            'presbylaryngis': {
                'acoustic_profile': {
                    'hnr_mean': (5, 15),
                    'breathiness_score': (1, 2),
                    'intensity_mean': (45, 70),
                    'f0_std': (2, 10),
                    'voice_breaks_percentage': (2, 15)
                },
                'perceptual_profile': {
                    'B': (1, 2),  # Mild-moderate breathiness
                    'A': (1, 2),  # Mild-moderate weakness
                    'G': (1, 2)   # Mild-moderate impairment
                },
                'spectrographic_signs': [
                    'mild harmonic irregularity',
                    'occasional voice breaks',
                    'reduced intensity'
                ],
                'confidence_weight': 0.6
            },
            
            'muscle_tension_dysphonia': {
                'acoustic_profile': {
                    'strain_score': (2, 3),
                    'very_hf_energy_ratio': (0.15, 1.0),
                    'spectral_tilt': (-5, 5),
                    'f0_mean': (150, 350)  # Often elevated
                },
                'perceptual_profile': {
                    'S': (2, 3),  # High strain
                    'G': (1, 3),  # Variable impairment
                    'R': (0, 2)   # Variable roughness
                },
                'spectrographic_signs': [
                    'high frequency emphasis',
                    'constricted formants',
                    'irregular harmonic spacing'
                ],
                'confidence_weight': 0.7
            },
            
            'vocal_nodules_polyps': {
                'acoustic_profile': {
                    'jitter_local': (0.02, 1.0),
                    'shimmer_local': (0.1, 1.0),
                    'hnr_mean': (5, 15),
                    'roughness_score': (1, 3)
                },
                'perceptual_profile': {
                    'R': (1, 3),  # Roughness prominent
                    'B': (0, 2),  # Variable breathiness
                    'G': (1, 3)   # Mild to severe
                },
                'spectrographic_signs': [
                    'subharmonics present',
                    'irregular voicing',
                    'harmonic perturbations'
                ],
                'confidence_weight': 0.8
            },
            
            'sulcus_vocalis': {
                'acoustic_profile': {
                    'hnr_mean': (3, 12),
                    'breathiness_score': (2, 3),
                    'roughness_score': (1, 2),
                    'intensity_mean': (40, 70)
                },
                'perceptual_profile': {
                    'B': (2, 3),  # Prominent breathiness
                    'R': (1, 2),  # Mild roughness
                    'A': (1, 2),  # Weakness
                    'G': (2, 3)   # Moderate to severe
                },
                'spectrographic_signs': [
                    'reduced harmonic amplitude',
                    'breath noise bands',
                    'inconsistent voicing'
                ],
                'confidence_weight': 0.7
            },
            
            'spasmodic_dysphonia': {
                'acoustic_profile': {
                    'voice_breaks_percentage': (10, 80),
                    'aperiodicity_score': (0.5, 1.0),
                    'strain_score': (2, 3),
                    'roughness_score': (2, 3)
                },
                'perceptual_profile': {
                    'S': (2, 3),  # High strain
                    'R': (2, 3),  # High roughness
                    'G': (2, 3)   # Severe impairment
                },
                'spectrographic_signs': [
                    'frequent voice breaks',
                    'irregular voicing patterns',
                    'abrupt onset/offset'
                ],
                'confidence_weight': 0.8
            }
        }
        
        # Triage recommendations
        self.triage_recommendations = {
            'urgent': {
                'conditions': ['suspected_malignancy', 'acute_trauma', 'airway_compromise'],
                'timeframe': 'within 48 hours',
                'specialist': 'ENT/Laryngology'
            },
            'soon': {
                'conditions': ['vocal_fold_paralysis', 'spasmodic_dysphonia', 'significant_lesions'],
                'timeframe': 'within 2-4 weeks',
                'specialist': 'ENT/Laryngology'
            },
            'routine': {
                'conditions': ['vocal_nodules_polyps', 'muscle_tension_dysphonia', 'presbylaryngis'],
                'timeframe': 'within 6-8 weeks',
                'specialist': 'ENT or SLP'
            },
            'conservative': {
                'conditions': ['mild_dysfunction', 'behavioral_issues'],
                'timeframe': 'voice therapy trial first',
                'specialist': 'SLP'
            }
        }
    
    def analyze_clinical_scenarios(self, 
                                 acoustic_features: Dict[str, float],
                                 grbas_results: Dict[str, any],
                                 spectrographic_features: Dict[str, any]) -> Dict[str, any]:
        """
        Analyze and map to clinical scenarios
        
        Args:
            acoustic_features: Acoustic analysis results
            grbas_results: GRBAS perceptual estimation results
            spectrographic_features: Spectrographic analysis results
            
        Returns:
            Dictionary with clinical impressions and recommendations
        """
        
        # Calculate pattern matches
        pattern_scores = self._calculate_pattern_matches(
            acoustic_features, grbas_results, spectrographic_features
        )
        
        # Determine primary and differential diagnoses
        primary_diagnosis, differential_diagnoses = self._determine_diagnoses(pattern_scores)
        
        # Generate clinical reasoning
        reasoning = self._generate_clinical_reasoning(
            primary_diagnosis, differential_diagnoses, 
            acoustic_features, grbas_results, spectrographic_features
        )
        
        # Determine triage level and recommendations
        triage_info = self._determine_triage(primary_diagnosis, grbas_results)
        
        # Generate next steps
        next_steps = self._generate_next_steps(primary_diagnosis, grbas_results, triage_info)
        
        return {
            'primary_impression': primary_diagnosis,
            'differential_diagnoses': differential_diagnoses,
            'pattern_matches': pattern_scores,
            'clinical_reasoning': reasoning,
            'triage': triage_info,
            'recommendations': next_steps,
            'red_flags': self._identify_red_flags(acoustic_features, grbas_results, spectrographic_features)
        }
    
    def _calculate_pattern_matches(self, 
                                 acoustic_features: Dict[str, float],
                                 grbas_results: Dict[str, any],
                                 spectrographic_features: Dict[str, any]) -> Dict[str, Dict[str, float]]:
        """Calculate how well patterns match clinical conditions"""
        
        pattern_scores = {}
        
        for condition, pattern in self.clinical_patterns.items():
            # Initialize scores
            acoustic_score = 0
            perceptual_score = 0
            spectrographic_score = 0
            total_features = 0
            
            # Acoustic pattern matching
            if 'acoustic_profile' in pattern:
                for feature, (min_val, max_val) in pattern['acoustic_profile'].items():
                    if feature in acoustic_features:
                        value = acoustic_features[feature]
                        if min_val <= value <= max_val:
                            acoustic_score += 1
                        total_features += 1
            
            # Perceptual pattern matching
            if 'perceptual_profile' in pattern:
                grbas_ratings = {
                    'G': grbas_results.get('G', 0),
                    'R': grbas_results.get('R', 0),
                    'B': grbas_results.get('B', 0),
                    'A': grbas_results.get('A', 0),
                    'S': grbas_results.get('S', 0)
                }
                
                for dimension, (min_val, max_val) in pattern['perceptual_profile'].items():
                    if dimension in grbas_ratings:
                        rating = grbas_ratings[dimension]
                        if min_val <= rating <= max_val:
                            perceptual_score += 1
                        total_features += 1
            
            # Spectrographic pattern matching (qualitative)
            if 'spectrographic_signs' in pattern:
                for sign in pattern['spectrographic_signs']:
                    if self._check_spectrographic_sign(sign, spectrographic_features):
                        spectrographic_score += 1
                    total_features += 1
            
            # Calculate overall match score
            if total_features > 0:
                raw_score = (acoustic_score + perceptual_score + spectrographic_score) / total_features
                confidence_weighted_score = raw_score * pattern.get('confidence_weight', 1.0)
                
                pattern_scores[condition] = {
                    'raw_score': raw_score,
                    'confidence_weighted_score': confidence_weighted_score,
                    'acoustic_match': acoustic_score / max(1, len(pattern.get('acoustic_profile', {}))),
                    'perceptual_match': perceptual_score / max(1, len(pattern.get('perceptual_profile', {}))),
                    'spectrographic_match': spectrographic_score / max(1, len(pattern.get('spectrographic_signs', [])))
                }
            else:
                pattern_scores[condition] = {
                    'raw_score': 0,
                    'confidence_weighted_score': 0,
                    'acoustic_match': 0,
                    'perceptual_match': 0,
                    'spectrographic_match': 0
                }
        
        return pattern_scores
    
    def _check_spectrographic_sign(self, sign: str, spectrographic_features: Dict[str, any]) -> bool:
        """Check if a spectrographic sign is present"""
        
        sign_checks = {
            'poor harmonic definition': lambda f: f.get('harmonic_definition', 'good') == 'poor',
            'elevated breath noise': lambda f: f.get('breath_noise_level', -60) > -30,
            'reduced harmonic clarity': lambda f: f.get('harmonic_clarity_score', 10) < 5,
            'subharmonic presence': lambda f: f.get('subharmonic_strength', 0) > 0.1,
            'irregular harmonic structure': lambda f: f.get('aperiodicity_score', 0) > 0.3,
            'aperiodic segments': lambda f: f.get('overall_aperiodicity', 'minimal') in ['moderate', 'severe'],
            'extensive breath noise': lambda f: f.get('breath_noise_rating', 'minimal') in ['moderate', 'severe'],
            'voice breaks': lambda f: f.get('voice_breaks_percentage', 0) > 5,
            'mild harmonic irregularity': lambda f: 0.1 < f.get('aperiodicity_score', 0) < 0.4,
            'occasional voice breaks': lambda f: 2 < f.get('voice_breaks_percentage', 0) < 10,
            'high frequency emphasis': lambda f: f.get('very_hf_energy_ratio', 0) > 0.2,
            'constricted formants': lambda f: f.get('formant_clarity', 'good') == 'poor',
            'irregular harmonic spacing': lambda f: f.get('harmonic_definition', 'good') == 'fair',
            'subharmonics present': lambda f: f.get('subharmonic_strength', 0) > 0.05,
            'harmonic perturbations': lambda f: f.get('aperiodicity_score', 0) > 0.2,
            'reduced harmonic amplitude': lambda f: f.get('harmonic_clarity_score', 10) < 3,
            'breath noise bands': lambda f: f.get('breath_noise_level', -60) > -40,
            'inconsistent voicing': lambda f: f.get('voice_breaks_percentage', 0) > 3,
            'frequent voice breaks': lambda f: f.get('voice_breaks_percentage', 0) > 15,
            'irregular voicing patterns': lambda f: f.get('overall_aperiodicity', 'minimal') == 'severe',
            'abrupt onset/offset': lambda f: f.get('voice_breaks_count', 0) > 5
        }
        
        check_function = sign_checks.get(sign)
        if check_function:
            return check_function(spectrographic_features)
        else:
            return False
    
    def _determine_diagnoses(self, pattern_scores: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, any], List[Dict[str, any]]]:
        """Determine primary and differential diagnoses"""
        
        # Sort by confidence-weighted score
        sorted_conditions = sorted(
            pattern_scores.items(),
            key=lambda x: x[1]['confidence_weighted_score'],
            reverse=True
        )
        
        if not sorted_conditions:
            return {}, []
        
        # Primary diagnosis (highest score)
        primary_condition, primary_scores = sorted_conditions[0]
        
        primary_diagnosis = {
            'condition': primary_condition,
            'confidence': primary_scores['confidence_weighted_score'],
            'pattern_strength': primary_scores['raw_score'],
            'description': self._get_condition_description(primary_condition)
        }
        
        # Differential diagnoses (other significant matches)
        differential_diagnoses = []
        for condition, scores in sorted_conditions[1:6]:  # Top 5 alternatives
            if scores['confidence_weighted_score'] > 0.3:  # Minimum threshold
                differential_diagnoses.append({
                    'condition': condition,
                    'confidence': scores['confidence_weighted_score'],
                    'pattern_strength': scores['raw_score'],
                    'description': self._get_condition_description(condition)
                })
        
        return primary_diagnosis, differential_diagnoses
    
    def _get_condition_description(self, condition: str) -> str:
        """Get clinical description of condition"""
        
        descriptions = {
            'glottic_insufficiency': 'Incomplete vocal fold closure resulting in breathy voice quality',
            'vocal_fold_lesions': 'Mass lesions on vocal folds causing irregular vibration and roughness',
            'vocal_fold_paralysis': 'Unilateral or bilateral vocal fold paralysis with severe breathiness',
            'presbylaryngis': 'Age-related vocal fold atrophy with mild breathiness and weakness',
            'muscle_tension_dysphonia': 'Excessive laryngeal muscle tension causing strained voice quality',
            'vocal_nodules_polyps': 'Benign vocal fold lesions causing roughness and breathiness',
            'sulcus_vocalis': 'Vocal fold scarring/sulcus causing breathiness and reduced projection',
            'spasmodic_dysphonia': 'Neurological voice disorder with involuntary vocal spasms'
        }
        
        return descriptions.get(condition, 'Voice disorder requiring further evaluation')
    
    def _generate_clinical_reasoning(self, 
                                   primary_diagnosis: Dict[str, any],
                                   differential_diagnoses: List[Dict[str, any]],
                                   acoustic_features: Dict[str, float],
                                   grbas_results: Dict[str, any],
                                   spectrographic_features: Dict[str, any]) -> str:
        """Generate clinical reasoning explanation"""
        
        reasoning_parts = []
        
        # Primary diagnosis reasoning
        if primary_diagnosis:
            condition = primary_diagnosis['condition']
            confidence = primary_diagnosis['confidence']
            
            reasoning_parts.append(
                f"Primary impression suggests {condition.replace('_', ' ')} "
                f"(confidence: {confidence:.1%}). "
            )
            
            # Key supporting features
            key_features = self._identify_key_supporting_features(
                condition, acoustic_features, grbas_results, spectrographic_features
            )
            if key_features:
                reasoning_parts.append(f"Key supporting features: {', '.join(key_features)}. ")
        
        # Differential considerations
        if differential_diagnoses:
            top_differential = differential_diagnoses[0]
            reasoning_parts.append(
                f"Main differential consideration is {top_differential['condition'].replace('_', ' ')} "
                f"(confidence: {top_differential['confidence']:.1%}). "
            )
        
        # Overall GRBAS pattern
        grbas_string = f"G{grbas_results.get('G', 0)}R{grbas_results.get('R', 0)}B{grbas_results.get('B', 0)}A{grbas_results.get('A', 0)}S{grbas_results.get('S', 0)}"
        reasoning_parts.append(f"GRBAS pattern: {grbas_string}. ")
        
        return ''.join(reasoning_parts)
    
    def _identify_key_supporting_features(self, 
                                        condition: str,
                                        acoustic_features: Dict[str, float],
                                        grbas_results: Dict[str, any],
                                        spectrographic_features: Dict[str, any]) -> List[str]:
        """Identify key features supporting the diagnosis"""
        
        supporting_features = []
        
        # Condition-specific key features
        if condition == 'glottic_insufficiency':
            if grbas_results.get('B', 0) >= 2:
                supporting_features.append('prominent breathiness')
            if acoustic_features.get('hnr_mean', 20) < 10:
                supporting_features.append('low HNR')
            if spectrographic_features.get('breath_noise_level', -60) > -30:
                supporting_features.append('elevated breath noise')
        
        elif condition == 'vocal_fold_lesions':
            if grbas_results.get('R', 0) >= 2:
                supporting_features.append('significant roughness')
            if acoustic_features.get('jitter_local', 0) > 0.02:
                supporting_features.append('elevated jitter')
            if spectrographic_features.get('subharmonic_strength', 0) > 0.1:
                supporting_features.append('subharmonics present')
        
        elif condition == 'muscle_tension_dysphonia':
            if grbas_results.get('S', 0) >= 2:
                supporting_features.append('vocal strain')
            if acoustic_features.get('very_hf_energy_ratio', 0) > 0.2:
                supporting_features.append('high-frequency emphasis')
        
        elif condition == 'vocal_fold_paralysis':
            if grbas_results.get('B', 0) >= 3:
                supporting_features.append('severe breathiness')
            if grbas_results.get('A', 0) >= 2:
                supporting_features.append('vocal weakness')
            if acoustic_features.get('intensity_mean', 80) < 65:
                supporting_features.append('reduced intensity')
        
        return supporting_features
    
    def _determine_triage(self, 
                         primary_diagnosis: Dict[str, any],
                         grbas_results: Dict[str, any]) -> Dict[str, any]:
        """Determine appropriate triage level"""
        
        if not primary_diagnosis:
            return {
                'level': 'routine',
                'timeframe': 'within 6-8 weeks',
                'specialist': 'ENT or SLP',
                'reasoning': 'Unclear diagnosis requires evaluation'
            }
        
        condition = primary_diagnosis['condition']
        confidence = primary_diagnosis['confidence']
        overall_severity = grbas_results.get('G', 0)
        
        # Urgent conditions
        if condition in ['vocal_fold_paralysis'] and confidence > 0.7:
            return {
                'level': 'soon',
                'timeframe': 'within 2-4 weeks',
                'specialist': 'ENT/Laryngology',
                'reasoning': 'Suspected vocal fold paralysis requires prompt laryngoscopy'
            }
        
        # Severe overall impairment
        if overall_severity >= 3:
            return {
                'level': 'soon',
                'timeframe': 'within 2-4 weeks',
                'specialist': 'ENT/Laryngology',
                'reasoning': 'Severe voice impairment requires medical evaluation'
            }
        
        # Suspected organic pathology
        if condition in ['vocal_fold_lesions', 'spasmodic_dysphonia'] and confidence > 0.6:
            return {
                'level': 'soon',
                'timeframe': 'within 2-4 weeks',
                'specialist': 'ENT/Laryngology',
                'reasoning': 'Suspected organic pathology requires visualization'
            }
        
        # Functional disorders
        if condition in ['muscle_tension_dysphonia'] and overall_severity <= 2:
            return {
                'level': 'conservative',
                'timeframe': 'voice therapy trial first',
                'specialist': 'SLP',
                'reasoning': 'Functional disorder may respond to voice therapy'
            }
        
        # Default to routine
        return {
            'level': 'routine',
            'timeframe': 'within 6-8 weeks',
            'specialist': 'ENT or SLP',
            'reasoning': 'Standard voice evaluation indicated'
        }
    
    def _generate_next_steps(self, 
                           primary_diagnosis: Dict[str, any],
                           grbas_results: Dict[str, any],
                           triage_info: Dict[str, any]) -> Dict[str, any]:
        """Generate specific next steps and recommendations"""
        
        next_steps = {
            'instrumental_evaluation': [],
            'therapy_recommendations': [],
            'lifestyle_modifications': [],
            'follow_up': []
        }
        
        if not primary_diagnosis:
            next_steps['instrumental_evaluation'].append('Comprehensive voice evaluation')
            return next_steps
        
        condition = primary_diagnosis['condition']
        
        # Instrumental evaluation recommendations
        if condition in ['vocal_fold_paralysis', 'vocal_fold_lesions', 'spasmodic_dysphonia']:
            next_steps['instrumental_evaluation'].extend([
                'Videostroboscopy for vocal fold visualization',
                'Consider airflow/pressure measurements if available'
            ])
        
        if grbas_results.get('A', 0) >= 2:  # Weakness
            next_steps['instrumental_evaluation'].extend([
                'Maximum phonation time assessment',
                's/z ratio measurement'
            ])
        
        # Therapy recommendations
        if condition == 'muscle_tension_dysphonia':
            next_steps['therapy_recommendations'].extend([
                'Vocal hygiene education',
                'Relaxation techniques for laryngeal tension',
                'Voice therapy focusing on efficient voice production'
            ])
        
        elif condition in ['vocal_nodules_polyps']:
            next_steps['therapy_recommendations'].extend([
                'Voice rest period if acute',
                'Vocal hygiene counseling',
                'Behavioral voice therapy'
            ])
        
        elif condition == 'presbylaryngis':
            next_steps['therapy_recommendations'].extend([
                'Vocal function exercises',
                'Respiratory support training'
            ])
        
        # Lifestyle modifications
        if grbas_results.get('B', 0) >= 2 or condition in ['glottic_insufficiency']:
            next_steps['lifestyle_modifications'].extend([
                'Avoid excessive voice use',
                'Stay well hydrated',
                'Consider humidification'
            ])
        
        # Follow-up recommendations
        if triage_info['level'] == 'soon':
            next_steps['follow_up'].append(f"Medical evaluation {triage_info['timeframe']}")
        
        if condition in ['vocal_nodules_polyps', 'muscle_tension_dysphonia']:
            next_steps['follow_up'].append('Re-evaluate after 6-8 weeks of therapy')
        
        return next_steps
    
    def _identify_red_flags(self, 
                          acoustic_features: Dict[str, float],
                          grbas_results: Dict[str, any],
                          spectrographic_features: Dict[str, any]) -> List[str]:
        """Identify red flag symptoms requiring urgent attention"""
        
        red_flags = []
        
        # Severe overall impairment
        if grbas_results.get('G', 0) >= 3:
            red_flags.append('Severe voice impairment')
        
        # Very poor acoustic measures suggesting significant pathology
        if acoustic_features.get('hnr_mean', 20) < 3:
            red_flags.append('Extremely poor harmonics-to-noise ratio')
        
        # Extensive voice breaks
        if spectrographic_features.get('voice_breaks_percentage', 0) > 30:
            red_flags.append('Extensive voice breaks suggesting severe dysfunction')
        
        # Very low intensity suggesting possible paralysis
        if acoustic_features.get('intensity_mean', 80) < 50:
            red_flags.append('Very low vocal intensity')
        
        # Poor voice quality with short duration (possible fatigue/weakness)
        if (grbas_results.get('G', 0) >= 2 and 
            acoustic_features.get('voiced_fraction', 1.0) < 0.5):
            red_flags.append('Poor voice quality with reduced voicing capability')
        
        return red_flags