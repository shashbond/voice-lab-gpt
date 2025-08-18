"""
Report generation for Voice Lab GPT
Generates clinical reports in PDF, HTML, and JSON formats
"""

import json
import datetime
from typing import Dict, List, Optional, Any
from jinja2 import Template
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import base64
from io import BytesIO
import warnings

class ReportGenerator:
    """Generates comprehensive clinical reports"""
    
    def __init__(self):
        """Initialize report generator"""
        
        # HTML template for reports
        self.html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Lab GPT Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; }
        .header { text-align: center; border-bottom: 2px solid #2c3e50; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin: 20px 0; }
        .section h2 { color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
        .grbas-table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        .grbas-table td, .grbas-table th { border: 1px solid #ddd; padding: 8px; text-align: center; }
        .grbas-table th { background-color: #f2f2f2; }
        .acoustic-table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        .acoustic-table td, .acoustic-table th { border: 1px solid #ddd; padding: 8px; }
        .acoustic-table th { background-color: #f8f9fa; }
        .confidence { font-size: 0.9em; color: #666; }
        .impression { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .recommendations { background-color: #f0f8f0; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .red-flag { background-color: #ffeaea; color: #d63031; padding: 10px; border-radius: 5px; margin: 10px 0; }
        .spectrogram { text-align: center; margin: 20px 0; }
        .footer { margin-top: 30px; text-align: center; font-size: 0.9em; color: #666; border-top: 1px solid #bdc3c7; padding-top: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Voice Lab GPT Analysis Report</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        {% if patient_info %}
        <p><strong>Patient/File:</strong> {{ patient_info }}</p>
        {% endif %}
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="impression">
            <h3>Primary Impression</h3>
            <p><strong>{{ clinical_results.primary_impression.condition|replace('_', ' ')|title }}:</strong> 
            {{ clinical_results.primary_impression.description }}</p>
            <p><strong>Confidence:</strong> {{ "%.0f"|format(clinical_results.primary_impression.confidence * 100) }}%</p>
        </div>
        
        {% if clinical_results.red_flags %}
        <div class="red-flag">
            <h3>⚠️ Red Flags</h3>
            <ul>
                {% for flag in clinical_results.red_flags %}
                <li>{{ flag }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>

    <div class="section">
        <h2>Perceptual Assessment (GRBAS)</h2>
        <table class="grbas-table">
            <tr>
                <th>G<br><small>Grade</small></th>
                <th>R<br><small>Roughness</small></th>
                <th>B<br><small>Breathiness</small></th>
                <th>A<br><small>Asthenia</small></th>
                <th>S<br><small>Strain</small></th>
            </tr>
            <tr>
                <td><strong>{{ grbas_results.G }}</strong><br><span class="confidence">{{ "%.0f"|format(grbas_results.confidence.G * 100) }}%</span></td>
                <td><strong>{{ grbas_results.R }}</strong><br><span class="confidence">{{ "%.0f"|format(grbas_results.confidence.R * 100) }}%</span></td>
                <td><strong>{{ grbas_results.B }}</strong><br><span class="confidence">{{ "%.0f"|format(grbas_results.confidence.B * 100) }}%</span></td>
                <td><strong>{{ grbas_results.A }}</strong><br><span class="confidence">{{ "%.0f"|format(grbas_results.confidence.A * 100) }}%</span></td>
                <td><strong>{{ grbas_results.S }}</strong><br><span class="confidence">{{ "%.0f"|format(grbas_results.confidence.S * 100) }}%</span></td>
            </tr>
        </table>
        <p><strong>Overall Confidence:</strong> {{ "%.0f"|format(grbas_results.confidence.overall * 100) }}%</p>
        
        <h3>Perceptual Reasoning</h3>
        {% for dimension, reasoning in grbas_results.reasoning.items() %}
        <p><strong>{{ dimension }}:</strong> {{ reasoning }}</p>
        {% endfor %}
    </div>

    <div class="section">
        <h2>Acoustic Measures</h2>
        <table class="acoustic-table">
            <tr><th>Measure</th><th>Value</th><th>Normal Range</th><th>Status</th></tr>
            <tr><td>F0 Mean</td><td>{{ "%.1f"|format(acoustic_features.f0_mean) }} Hz</td><td>80-250 Hz</td><td>{{ get_status(acoustic_features.f0_mean, 80, 250) }}</td></tr>
            <tr><td>Jitter (local)</td><td>{{ "%.3f"|format(acoustic_features.jitter_local) }}</td><td>&lt; 0.01</td><td>{{ get_jitter_status(acoustic_features.jitter_local) }}</td></tr>
            <tr><td>Shimmer (local)</td><td>{{ "%.3f"|format(acoustic_features.shimmer_local) }}</td><td>&lt; 0.05</td><td>{{ get_shimmer_status(acoustic_features.shimmer_local) }}</td></tr>
            <tr><td>HNR</td><td>{{ "%.1f"|format(acoustic_features.hnr_mean) }} dB</td><td>&gt; 15 dB</td><td>{{ get_hnr_status(acoustic_features.hnr_mean) }}</td></tr>
            <tr><td>CPP</td><td>{{ "%.1f"|format(acoustic_features.cpp_mean) }} dB</td><td>&gt; 10 dB</td><td>{{ get_cpp_status(acoustic_features.cpp_mean) }}</td></tr>
            <tr><td>Spectral Tilt</td><td>{{ "%.1f"|format(acoustic_features.spectral_tilt) }} dB/oct</td><td>-12 to -6 dB/oct</td><td>{{ get_status(acoustic_features.spectral_tilt, -12, -6) }}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Spectrographic Findings</h2>
        <h3>Harmonic Structure</h3>
        <p><strong>Harmonic Definition:</strong> {{ spectrographic_features.harmonic_definition|title }}</p>
        <p><strong>Formant Clarity:</strong> {{ spectrographic_features.formant_clarity|title }}</p>
        <p><strong>Harmonic Clarity Score:</strong> {{ "%.1f"|format(spectrographic_features.harmonic_clarity_score) }}</p>
        
        <h3>Noise Characteristics</h3>
        <p><strong>Breath Noise Level:</strong> {{ "%.1f"|format(spectrographic_features.breath_noise_level) }} dB</p>
        <p><strong>Noise Floor:</strong> {{ "%.1f"|format(spectrographic_features.noise_floor_db) }} dB</p>
        
        <h3>Aperiodicity</h3>
        <p><strong>Overall Aperiodicity:</strong> {{ spectrographic_features.overall_aperiodicity|title }}</p>
        <p><strong>Subharmonic Strength:</strong> {{ "%.2f"|format(spectrographic_features.subharmonic_strength) }}</p>
        {% if spectrographic_features.subharmonic_frequency > 0 %}
        <p><strong>Subharmonic Frequency:</strong> {{ "%.1f"|format(spectrographic_features.subharmonic_frequency) }} Hz</p>
        {% endif %}
    </div>

    <div class="section">
        <h2>Clinical Reasoning</h2>
        <p>{{ clinical_results.clinical_reasoning }}</p>
        
        {% if clinical_results.differential_diagnoses %}
        <h3>Differential Diagnoses</h3>
        <ul>
            {% for diff in clinical_results.differential_diagnoses %}
            <li><strong>{{ diff.condition|replace('_', ' ')|title }}:</strong> {{ diff.description }} 
                <span class="confidence">({{ "%.0f"|format(diff.confidence * 100) }}% confidence)</span></li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <div class="recommendations">
            <h3>Triage: {{ clinical_results.triage.level|title }}</h3>
            <p><strong>Timeframe:</strong> {{ clinical_results.triage.timeframe }}</p>
            <p><strong>Specialist:</strong> {{ clinical_results.triage.specialist }}</p>
            <p><strong>Reasoning:</strong> {{ clinical_results.triage.reasoning }}</p>
        </div>
        
        {% if clinical_results.recommendations.instrumental_evaluation %}
        <h3>Instrumental Evaluation</h3>
        <ul>
            {% for rec in clinical_results.recommendations.instrumental_evaluation %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        
        {% if clinical_results.recommendations.therapy_recommendations %}
        <h3>Therapy Recommendations</h3>
        <ul>
            {% for rec in clinical_results.recommendations.therapy_recommendations %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        
        {% if clinical_results.recommendations.lifestyle_modifications %}
        <h3>Lifestyle Modifications</h3>
        <ul>
            {% for rec in clinical_results.recommendations.lifestyle_modifications %}
            <li>{{ rec }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>

    <div class="footer">
        <p><strong>🤖 Generated with Voice Lab GPT</strong></p>
        <p>This automated analysis should be interpreted by qualified voice professionals. 
        Clinical correlation and direct examination are essential for diagnosis and treatment planning.</p>
    </div>
</body>
</html>
        """
        
    def generate_complete_report(self, 
                               acoustic_features: Dict[str, float],
                               grbas_results: Dict[str, any],
                               spectrographic_features: Dict[str, any],
                               clinical_results: Dict[str, any],
                               patient_info: Optional[str] = None,
                               include_plots: bool = True) -> Dict[str, any]:
        """
        Generate complete report in multiple formats
        
        Args:
            acoustic_features: Acoustic analysis results
            grbas_results: GRBAS estimation results
            spectrographic_features: Spectrographic analysis results
            clinical_results: Clinical scenario mapping results
            patient_info: Patient or file identifier
            include_plots: Whether to include visualization plots
            
        Returns:
            Dictionary with reports in different formats
        """
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create comprehensive data structure
        report_data = {
            'metadata': {
                'timestamp': timestamp,
                'patient_info': patient_info or 'Unknown',
                'analysis_version': '1.0.0'
            },
            'acoustic_features': acoustic_features,
            'grbas_results': grbas_results,
            'spectrographic_features': spectrographic_features,
            'clinical_results': clinical_results
        }
        
        # Generate different report formats
        reports = {}
        
        # JSON report
        reports['json'] = self._generate_json_report(report_data)
        
        # HTML report
        reports['html'] = self._generate_html_report(report_data)
        
        # Clinical summary (concise text)
        reports['clinical_summary'] = self._generate_clinical_summary(report_data)
        
        # Detailed text report
        reports['detailed_report'] = self._generate_detailed_text_report(report_data)
        
        # Generate plots if requested
        if include_plots:
            plots = self._generate_plots(acoustic_features, spectrographic_features)
            reports['plots'] = plots
        
        return reports
    
    def _generate_json_report(self, report_data: Dict[str, any]) -> str:
        """Generate structured JSON report for EMR/analytics"""
        
        # Clean up data for JSON serialization
        json_data = self._clean_for_json(report_data.copy())
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
    
    def _clean_for_json(self, obj):
        """Recursively clean data for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _generate_html_report(self, report_data: Dict[str, any]) -> str:
        """Generate comprehensive HTML report"""
        
        # Create Jinja2 template
        template = Template(self.html_template)
        
        # Helper functions for template
        def get_status(value, min_normal, max_normal):
            if min_normal <= value <= max_normal:
                return "✓ Normal"
            elif value < min_normal:
                return "↓ Below Normal"
            else:
                return "↑ Above Normal"
        
        def get_jitter_status(value):
            if value <= 0.01:
                return "✓ Normal"
            elif value <= 0.02:
                return "⚠ Elevated"
            else:
                return "❌ High"
        
        def get_shimmer_status(value):
            if value <= 0.05:
                return "✓ Normal"
            elif value <= 0.08:
                return "⚠ Elevated"
            else:
                return "❌ High"
        
        def get_hnr_status(value):
            if value >= 15:
                return "✓ Normal"
            elif value >= 10:
                return "⚠ Reduced"
            else:
                return "❌ Poor"
        
        def get_cpp_status(value):
            if value >= 10:
                return "✓ Normal"
            elif value >= 5:
                return "⚠ Reduced"
            else:
                return "❌ Poor"
        
        # Render template
        try:
            html_content = template.render(
                timestamp=report_data['metadata']['timestamp'],
                patient_info=report_data['metadata']['patient_info'],
                acoustic_features=report_data['acoustic_features'],
                grbas_results=report_data['grbas_results'],
                spectrographic_features=report_data['spectrographic_features'],
                clinical_results=report_data['clinical_results'],
                get_status=get_status,
                get_jitter_status=get_jitter_status,
                get_shimmer_status=get_shimmer_status,
                get_hnr_status=get_hnr_status,
                get_cpp_status=get_cpp_status
            )
            return html_content
        except Exception as e:
            warnings.warn(f"HTML generation failed: {str(e)}")
            return self._generate_fallback_html(report_data)
    
    def _generate_fallback_html(self, report_data: Dict[str, any]) -> str:
        """Generate simple fallback HTML if template fails"""
        
        html = f"""
        <html>
        <body>
            <h1>Voice Lab GPT Report</h1>
            <p><strong>Generated:</strong> {report_data['metadata']['timestamp']}</p>
            <p><strong>Patient/File:</strong> {report_data['metadata']['patient_info']}</p>
            
            <h2>GRBAS Results</h2>
            <p>G={report_data['grbas_results']['G']}, R={report_data['grbas_results']['R']}, 
               B={report_data['grbas_results']['B']}, A={report_data['grbas_results']['A']}, 
               S={report_data['grbas_results']['S']}</p>
            
            <h2>Primary Impression</h2>
            <p>{report_data['clinical_results']['primary_impression']['condition'].replace('_', ' ').title()}</p>
            <p>Confidence: {report_data['clinical_results']['primary_impression']['confidence']:.1%}</p>
            
            <h2>Key Acoustic Measures</h2>
            <p>F0: {report_data['acoustic_features']['f0_mean']:.1f} Hz</p>
            <p>Jitter: {report_data['acoustic_features']['jitter_local']:.3f}</p>
            <p>Shimmer: {report_data['acoustic_features']['shimmer_local']:.3f}</p>
            <p>HNR: {report_data['acoustic_features']['hnr_mean']:.1f} dB</p>
        </body>
        </html>
        """
        return html
    
    def _generate_clinical_summary(self, report_data: Dict[str, any]) -> str:
        """Generate concise clinical summary"""
        
        grbas = report_data['grbas_results']
        primary = report_data['clinical_results']['primary_impression']
        triage = report_data['clinical_results']['triage']
        
        summary_lines = [
            f"Voice Lab GPT Analysis - {report_data['metadata']['timestamp']}",
            f"Patient/File: {report_data['metadata']['patient_info']}",
            "",
            f"GRBAS: G={grbas['G']}, R={grbas['R']}, B={grbas['B']}, A={grbas['A']}, S={grbas['S']} "
            f"(Confidence: {grbas['confidence']['overall']:.0%})",
            "",
            f"PRIMARY IMPRESSION: {primary['condition'].replace('_', ' ').title()} "
            f"(Confidence: {primary['confidence']:.0%})",
            f"{primary['description']}",
            "",
            f"KEY ACOUSTICS:",
            f"  F0: {report_data['acoustic_features']['f0_mean']:.1f} Hz, "
            f"Jitter: {report_data['acoustic_features']['jitter_local']:.3f}, "
            f"Shimmer: {report_data['acoustic_features']['shimmer_local']:.3f}",
            f"  HNR: {report_data['acoustic_features']['hnr_mean']:.1f} dB, "
            f"CPP: {report_data['acoustic_features']['cpp_mean']:.1f} dB, "
            f"Spectral Tilt: {report_data['acoustic_features']['spectral_tilt']:.1f} dB/oct",
            "",
            f"TRIAGE: {triage['level'].upper()} - {triage['timeframe']}",
            f"Specialist: {triage['specialist']}",
            f"Reasoning: {triage['reasoning']}"
        ]
        
        # Add red flags if present
        if report_data['clinical_results']['red_flags']:
            summary_lines.extend([
                "",
                "⚠️  RED FLAGS:",
                *[f"  • {flag}" for flag in report_data['clinical_results']['red_flags']]
            ])
        
        return "\n".join(summary_lines)
    
    def _generate_detailed_text_report(self, report_data: Dict[str, any]) -> str:
        """Generate detailed text report (PRAAT/clinic style)"""
        
        lines = [
            "="*60,
            "VOICE LAB GPT - COMPREHENSIVE ANALYSIS REPORT",
            "="*60,
            "",
            f"Generated: {report_data['metadata']['timestamp']}",
            f"Patient/File: {report_data['metadata']['patient_info']}",
            f"Analysis Version: {report_data['metadata']['analysis_version']}",
            "",
            "1. PERCEPTUAL ASSESSMENT (GRBAS)",
            "-"*40
        ]
        
        # GRBAS section
        grbas = report_data['grbas_results']
        lines.extend([
            f"G (Grade):      {grbas['G']}/3 (Confidence: {grbas['confidence']['G']:.0%})",
            f"R (Roughness):  {grbas['R']}/3 (Confidence: {grbas['confidence']['R']:.0%})",
            f"B (Breathiness): {grbas['B']}/3 (Confidence: {grbas['confidence']['B']:.0%})",
            f"A (Asthenia):   {grbas['A']}/3 (Confidence: {grbas['confidence']['A']:.0%})",
            f"S (Strain):     {grbas['S']}/3 (Confidence: {grbas['confidence']['S']:.0%})",
            f"Overall Confidence: {grbas['confidence']['overall']:.0%}",
            ""
        ])
        
        # Reasoning
        lines.extend([
            "Perceptual Reasoning:",
            *[f"  {dim}: {reason}" for dim, reason in grbas['reasoning'].items()],
            ""
        ])
        
        # Acoustic measures
        lines.extend([
            "2. ACOUSTIC MEASURES",
            "-"*40
        ])
        
        acoustic = report_data['acoustic_features']
        lines.extend([
            f"Fundamental Frequency:",
            f"  Mean: {acoustic['f0_mean']:.1f} Hz",
            f"  SD: {acoustic['f0_std']:.1f} Hz",
            f"  Range: {acoustic['f0_min']:.1f} - {acoustic['f0_max']:.1f} Hz",
            f"  Voiced fraction: {acoustic['voiced_fraction']:.2f}",
            "",
            f"Perturbation Measures:",
            f"  Jitter (local): {acoustic['jitter_local']:.4f} ({self._get_jitter_interpretation(acoustic['jitter_local'])})",
            f"  Jitter (RAP): {acoustic['jitter_rap']:.4f}",
            f"  Shimmer (local): {acoustic['shimmer_local']:.4f} ({self._get_shimmer_interpretation(acoustic['shimmer_local'])})",
            f"  Shimmer (APQ11): {acoustic['shimmer_apq11']:.4f}",
            "",
            f"Harmonic Measures:",
            f"  HNR: {acoustic['hnr_mean']:.1f} dB ({self._get_hnr_interpretation(acoustic['hnr_mean'])})",
            f"  NHR: {acoustic.get('nhr_mean', 0):.3f}",
            "",
            f"Cepstral Measures:",
            f"  CPP: {acoustic['cpp_mean']:.1f} dB ({self._get_cpp_interpretation(acoustic['cpp_mean'])})",
            f"  CPPS: {acoustic['cpps_mean']:.1f} dB",
            "",
            f"Spectral Measures:",
            f"  Spectral Tilt: {acoustic['spectral_tilt']:.1f} dB/oct",
            f"  Spectral Centroid: {acoustic.get('spectral_centroid', 0):.0f} Hz",
            f"  High-freq Ratio: {acoustic['high_freq_ratio']:.3f}",
            "",
            f"Intensity:",
            f"  Mean: {acoustic['intensity_mean']:.1f} dB",
            f"  Range: {acoustic['intensity_range']:.1f} dB",
            ""
        ])
        
        # Spectrographic findings
        lines.extend([
            "3. SPECTROGRAPHIC FINDINGS",
            "-"*40
        ])
        
        spectro = report_data['spectrographic_features']
        lines.extend([
            f"Harmonic Structure:",
            f"  Definition: {spectro['harmonic_definition'].title()}",
            f"  Clarity Score: {spectro['harmonic_clarity_score']:.1f}",
            f"  Formant Clarity: {spectro['formant_clarity'].title()}",
            "",
            f"Noise Characteristics:",
            f"  Breath Noise Level: {spectro['breath_noise_level']:.1f} dB",
            f"  Noise Floor: {spectro['noise_floor_db']:.1f} dB",
            f"  High-freq Noise Ratio: {spectro['high_freq_noise_ratio']:.3f}",
            "",
            f"Aperiodicity:",
            f"  Overall Level: {spectro['overall_aperiodicity'].title()}",
            f"  Aperiodicity Score: {spectro['aperiodicity_score']:.2f}",
            f"  Subharmonic Strength: {spectro['subharmonic_strength']:.3f}",
        ])
        
        if spectro.get('subharmonic_frequency', 0) > 0:
            lines.append(f"  Subharmonic Frequency: {spectro['subharmonic_frequency']:.1f} Hz")
        
        lines.extend([
            f"  Voice Breaks: {spectro['voice_breaks_percentage']:.1f}%",
            ""
        ])
        
        # Clinical impression
        lines.extend([
            "4. CLINICAL IMPRESSION & DIFFERENTIAL",
            "-"*40
        ])
        
        clinical = report_data['clinical_results']
        primary = clinical['primary_impression']
        lines.extend([
            f"Primary Impression: {primary['condition'].replace('_', ' ').title()}",
            f"  Description: {primary['description']}",
            f"  Confidence: {primary['confidence']:.0%}",
            f"  Pattern Strength: {primary['pattern_strength']:.2f}",
            ""
        ])
        
        if clinical.get('differential_diagnoses'):
            lines.append("Differential Diagnoses:")
            for i, diff in enumerate(clinical['differential_diagnoses'][:3], 1):
                lines.append(f"  {i}. {diff['condition'].replace('_', ' ').title()} ({diff['confidence']:.0%})")
            lines.append("")
        
        lines.extend([
            "Clinical Reasoning:",
            f"  {clinical['clinical_reasoning']}",
            ""
        ])
        
        # Recommendations
        lines.extend([
            "5. RECOMMENDATIONS",
            "-"*40
        ])
        
        triage = clinical['triage']
        lines.extend([
            f"Triage Level: {triage['level'].upper()}",
            f"  Timeframe: {triage['timeframe']}",
            f"  Specialist: {triage['specialist']}",
            f"  Reasoning: {triage['reasoning']}",
            ""
        ])
        
        recommendations = clinical['recommendations']
        if recommendations.get('instrumental_evaluation'):
            lines.append("Instrumental Evaluation:")
            for rec in recommendations['instrumental_evaluation']:
                lines.append(f"  • {rec}")
            lines.append("")
        
        if recommendations.get('therapy_recommendations'):
            lines.append("Therapy Recommendations:")
            for rec in recommendations['therapy_recommendations']:
                lines.append(f"  • {rec}")
            lines.append("")
        
        if recommendations.get('lifestyle_modifications'):
            lines.append("Lifestyle Modifications:")
            for rec in recommendations['lifestyle_modifications']:
                lines.append(f"  • {rec}")
            lines.append("")
        
        # Red flags
        if clinical.get('red_flags'):
            lines.extend([
                "⚠️  RED FLAGS:",
                *[f"  • {flag}" for flag in clinical['red_flags']],
                ""
            ])
        
        # Footer
        lines.extend([
            "="*60,
            "🤖 Generated with Voice Lab GPT",
            "This automated analysis requires clinical interpretation",
            "and correlation with direct examination.",
            "="*60
        ])
        
        return "\n".join(lines)
    
    def _get_jitter_interpretation(self, value: float) -> str:
        """Get interpretation for jitter value"""
        if value <= 0.01:
            return "Normal"
        elif value <= 0.02:
            return "Mildly elevated"
        elif value <= 0.04:
            return "Moderately elevated"
        else:
            return "Severely elevated"
    
    def _get_shimmer_interpretation(self, value: float) -> str:
        """Get interpretation for shimmer value"""
        if value <= 0.05:
            return "Normal"
        elif value <= 0.08:
            return "Mildly elevated"
        elif value <= 0.12:
            return "Moderately elevated"
        else:
            return "Severely elevated"
    
    def _get_hnr_interpretation(self, value: float) -> str:
        """Get interpretation for HNR value"""
        if value >= 15:
            return "Normal"
        elif value >= 10:
            return "Mildly reduced"
        elif value >= 5:
            return "Moderately reduced"
        else:
            return "Severely reduced"
    
    def _get_cpp_interpretation(self, value: float) -> str:
        """Get interpretation for CPP value"""
        if value >= 15:
            return "Excellent"
        elif value >= 10:
            return "Good"
        elif value >= 5:
            return "Reduced"
        else:
            return "Poor"
    
    def _generate_plots(self, 
                       acoustic_features: Dict[str, float], 
                       spectrographic_features: Dict[str, any]) -> Dict[str, str]:
        """Generate visualization plots as base64 encoded images"""
        
        plots = {}
        
        try:
            # GRBAS visualization
            plots['grbas_plot'] = self._create_grbas_plot(acoustic_features)
            
            # Acoustic measures radar chart
            plots['acoustic_radar'] = self._create_acoustic_radar(acoustic_features)
            
            # Feature summary bar chart
            plots['feature_summary'] = self._create_feature_summary(acoustic_features)
            
        except Exception as e:
            warnings.warn(f"Plot generation failed: {str(e)}")
            plots['error'] = f"Plot generation failed: {str(e)}"
        
        return plots
    
    def _create_grbas_plot(self, acoustic_features: Dict[str, float]) -> str:
        """Create GRBAS visualization"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Estimate GRBAS from acoustic features (simplified)
        grbas_values = [
            min(3, max(0, 3 - acoustic_features.get('hnr_mean', 15) / 5)),  # G
            min(3, acoustic_features.get('jitter_local', 0) * 150),  # R
            min(3, max(0, 3 - acoustic_features.get('hnr_mean', 15) / 5)),  # B
            min(3, max(0, (70 - acoustic_features.get('intensity_mean', 70)) / 10)),  # A
            min(3, acoustic_features.get('very_hf_energy_ratio', 0) * 10)  # S
        ]
        
        dimensions = ['Grade', 'Roughness', 'Breathiness', 'Asthenia', 'Strain']
        
        bars = ax.bar(dimensions, grbas_values, color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#2ecc71'])
        
        ax.set_ylim(0, 3)
        ax.set_ylabel('GRBAS Rating (0-3)')
        ax.set_title('GRBAS Perceptual Estimation', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, grbas_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode('utf-8')
    
    def _create_acoustic_radar(self, acoustic_features: Dict[str, float]) -> str:
        """Create acoustic measures radar chart"""
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Normalize features to 0-1 scale for radar chart
        features = {
            'F0 Stability': 1 - min(1, acoustic_features.get('f0_std', 0) / 50),
            'Voice Quality (HNR)': min(1, acoustic_features.get('hnr_mean', 0) / 20),
            'Periodicity': 1 - min(1, acoustic_features.get('jitter_local', 0) * 100),
            'Amplitude Stability': 1 - min(1, acoustic_features.get('shimmer_local', 0) * 20),
            'Cepstral Prominence': min(1, acoustic_features.get('cpp_mean', 0) / 20),
            'Vocal Intensity': min(1, max(0, acoustic_features.get('intensity_mean', 60) - 40) / 40)
        }
        
        labels = list(features.keys())
        values = list(features.values())
        
        # Add first value to end to close the polygon
        values += values[:1]
        
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label='Current Voice', color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        
        # Add reference circle for normal range
        normal_values = [0.8] * len(angles)
        ax.plot(angles, normal_values, '--', linewidth=1, color='green', alpha=0.7, label='Normal Range')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title('Acoustic Profile', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode('utf-8')
    
    def _create_feature_summary(self, acoustic_features: Dict[str, float]) -> str:
        """Create feature summary bar chart"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Key acoustic measures
        measures = ['F0 (Hz)', 'Jitter', 'Shimmer', 'HNR (dB)', 'CPP (dB)']
        values = [
            acoustic_features.get('f0_mean', 0),
            acoustic_features.get('jitter_local', 0) * 1000,  # Convert to per mille
            acoustic_features.get('shimmer_local', 0) * 100,  # Convert to percentage
            acoustic_features.get('hnr_mean', 0),
            acoustic_features.get('cpp_mean', 0)
        ]
        
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
        bars1 = ax1.bar(measures, values, color=colors)
        
        ax1.set_title('Key Acoustic Measures')
        ax1.set_ylabel('Value')
        
        # Add value labels
        for bar, value in zip(bars1, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Voice quality indicators
        quality_measures = ['Vocal Stability', 'Harmonic Clarity', 'Spectral Balance']
        quality_values = [
            min(100, max(0, 100 - acoustic_features.get('jitter_local', 0) * 5000)),
            min(100, acoustic_features.get('hnr_mean', 0) * 5),
            min(100, max(0, 50 + acoustic_features.get('spectral_tilt', -10)))
        ]
        
        bars2 = ax2.bar(quality_measures, quality_values, color=['#2ecc71', '#3498db', '#f39c12'])
        
        ax2.set_title('Voice Quality Indicators')
        ax2.set_ylabel('Quality Score (%)')
        ax2.set_ylim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars2, quality_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{value:.0f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode('utf-8')