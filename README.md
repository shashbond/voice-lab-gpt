# Voice Lab GPT 🎙️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/voice-lab-gpt/blob/main/Voice_Lab_GPT_Quick_Start.ipynb)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://hub.docker.com/)

**Professional Voice and Speech Analysis System**

Voice Lab GPT is a comprehensive, professional-grade voice analysis system that provides acoustic measurements, perceptual assessments, clinical scenario mapping, and automated reporting for voice professionals, researchers, and clinicians.

## Features ✨

### Core Analysis Capabilities
- **Acoustic Measures**: F0, jitter, shimmer, HNR, CPP, NHR, spectral tilt, intensity, formants
- **Spectrographic Analysis**: Harmonic definition, noise characteristics, aperiodicity, voice breaks
- **Perceptual Assessment**: GRBAS (Grade, Roughness, Breathiness, Asthenia, Strain) estimation
- **Clinical Mapping**: Automated mapping to clinical scenarios and differential diagnoses
- **Quality Assessment**: Automatic analysis reliability and signal quality evaluation

### Professional Reporting
- **Multiple Formats**: JSON, HTML, PDF-ready, clinical summaries
- **Comprehensive Visualizations**: Spectrograms, acoustic dashboards, GRBAS profiles
- **Clinical Reasoning**: Detailed explanations for findings and recommendations
- **Triage Recommendations**: Appropriate referral timeframes and specialists

### Clinical Applications
- Voice disorder assessment and monitoring
- Pre/post treatment comparison
- Research and documentation
- Telepractice and remote assessment
- Educational demonstrations

## Quick Start 🚀

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/voice-lab-gpt.git
cd voice-lab-gpt

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from voice_lab import VoiceLabGPT

# Initialize Voice Lab GPT
voice_lab = VoiceLabGPT()

# Analyze an audio file
results = voice_lab.analyze_file(
    'path/to/voice_sample.wav',
    voice_type='female',  # or 'male', None for auto
    patient_info='Patient ID: 12345',
    task_description='Sustained /a/ vowel'
)

# Get clinical summary
summary = voice_lab.get_clinical_summary()
print(summary)

# Get GRBAS ratings
grbas_string = voice_lab.get_grbas_string()  # e.g., "G2R1B3A1S0"
print(f"GRBAS: {grbas_string}")

# Save reports
voice_lab.save_reports(
    output_dir='./reports',
    formats=['html', 'json', 'clinical_summary']
)
```

### Analyze Audio Array

```python
import numpy as np

# Analyze audio data directly
audio_data = np.array([...])  # Your audio data
results = voice_lab.analyze_audio_array(
    audio_data,
    voice_type='male',
    patient_info='Research Participant 001'
)
```

## System Architecture 🏗️

Voice Lab GPT consists of several specialized components:

### 1. Audio Processor (`AudioProcessor`)
- Audio loading and format conversion
- Loudness normalization (ITU-R BS.1770)
- Silence removal and quality assessment
- Signal conditioning and preprocessing

### 2. Acoustic Analyzer (`AcousticAnalyzer`)
- Fundamental frequency analysis (F0)
- Perturbation measures (jitter, shimmer)
- Harmonics-to-noise ratio (HNR)
- Cepstral peak prominence (CPP)
- Spectral characteristics and formants

### 3. Spectrographic Analyzer (`SpectrographicAnalyzer`)
- Wideband spectrogram analysis
- Harmonic structure assessment
- Noise pattern identification
- Aperiodicity and subharmonic detection

### 4. GRBAS Estimator (`GRBASEstimator`)
- Rule-based perceptual assessment
- Multi-dimensional voice quality rating
- Confidence scoring and reasoning
- Clinical interpretation guidelines

### 5. Clinical Mapper (`ClinicalMapper`)
- Pattern matching to clinical conditions
- Differential diagnosis suggestions
- Triage level recommendations
- Evidence-based reasoning

### 6. Report Generator (`ReportGenerator`)
- Multi-format report generation
- Clinical summary creation
- Professional documentation
- Visualization integration

## Analysis Pipeline 📊

1. **Audio Preprocessing**
   - Load and convert audio file
   - Normalize loudness and remove silence
   - Assess signal quality

2. **Feature Extraction**
   - Extract acoustic measures
   - Perform spectrographic analysis
   - Calculate voice quality indices

3. **Perceptual Assessment**
   - Estimate GRBAS ratings
   - Calculate confidence scores
   - Generate reasoning explanations

4. **Clinical Analysis**
   - Map patterns to clinical scenarios
   - Determine differential diagnoses
   - Assess triage requirements

5. **Report Generation**
   - Create comprehensive reports
   - Generate visualizations
   - Provide clinical summaries

## Clinical Scenarios Supported 🏥

- **Glottic Insufficiency**: Incomplete vocal fold closure
- **Vocal Fold Lesions**: Nodules, polyps, and mass lesions
- **Vocal Fold Paralysis**: Unilateral/bilateral paralysis
- **Presbylaryngis**: Age-related vocal changes
- **Muscle Tension Dysphonia**: Functional voice disorders
- **Sulcus Vocalis**: Vocal fold scarring/sulcus
- **Spasmodic Dysphonia**: Neurological voice disorders

## GRBAS Assessment Scale 📏

| Rating | G (Grade) | R (Roughness) | B (Breathiness) | A (Asthenia) | S (Strain) |
|--------|-----------|---------------|----------------|--------------|------------|
| 0 | Normal | No roughness | No breathiness | Normal strength | No strain |
| 1 | Mild | Mild roughness | Mild breathiness | Mild weakness | Mild strain |
| 2 | Moderate | Moderate roughness | Moderate breathiness | Moderate weakness | Moderate strain |
| 3 | Severe | Severe roughness | Severe breathiness | Severe weakness | Severe strain |

## Quality Assurance 🔍

Voice Lab GPT includes comprehensive quality assessment:

- **Audio Quality**: SNR, clipping, duration, loudness
- **Analysis Reliability**: Feature extraction success, confidence levels
- **Clinical Validity**: Pattern matching strength, reasoning quality

Quality ratings:
- **Excellent**: >80% reliability
- **Good**: 60-80% reliability  
- **Fair**: 40-60% reliability
- **Poor**: <40% reliability

## Research Applications 🔬

### Academic Research
- Voice disorder characterization
- Treatment outcome measurement
- Population studies
- Method validation

### Clinical Research
- Therapy efficacy assessment
- Biomarker discovery
- Objective outcome measures
- Multi-site studies

## API Reference 📚

### VoiceLabGPT Class

#### Methods

**`analyze_file(file_path, voice_type=None, patient_info=None, task_description=None)`**
- Analyze audio file
- Returns comprehensive results dictionary

**`analyze_audio_array(audio, voice_type=None, patient_info=None, task_description=None)`**
- Analyze audio data array
- Returns comprehensive results dictionary

**`get_clinical_summary(results=None)`**
- Get concise clinical summary
- Returns formatted summary string

**`get_grbas_string(results=None)`**
- Get GRBAS rating string
- Returns string like "G1R2B0A1S0"

**`save_reports(output_dir, results=None, formats=['html', 'json'])`**
- Save analysis reports to files
- Returns dictionary of saved file paths

## Configuration ⚙️

### Initialization Parameters

```python
voice_lab = VoiceLabGPT(
    sr=16000,                    # Target sampling rate
    target_lufs=-23.0,           # Loudness normalization level
    silence_threshold_db=-40.0,  # Silence removal threshold
    enable_visualizations=True   # Generate plots
)
```

### Analysis Parameters

```python
results = voice_lab.analyze_file(
    'audio.wav',
    voice_type='female',          # 'male', 'female', None
    patient_info='Patient ID',    # Identifier/description
    task_description='Task',      # Recording task description
    generate_reports=True,        # Generate formatted reports
    generate_visualizations=True  # Generate plots
)
```

## Output Formats 📄

### JSON Report
Structured data for EMR integration and analytics:
```json
{
  "metadata": {...},
  "acoustic_features": {...},
  "grbas_results": {...},
  "clinical_results": {...},
  "quality_assessment": {...}
}
```

### HTML Report
Comprehensive clinical report with visualizations

### Clinical Summary
Concise text summary for clinical notes:
```
GRBAS: G1R2B0A1S0 (Confidence: 85%)
Primary Impression: Vocal Fold Lesions (Confidence: 78%)
Triage: SOON - within 2-4 weeks, ENT/Laryngology
```

## Examples 📝

See the `examples/` directory for:
- Basic usage examples
- Clinical workflow integration
- Research applications
- Custom analysis pipelines

## Testing 🧪

Run the test suite:
```bash
python -m pytest tests/
# or
python tests/test_voice_lab.py
```

Performance testing:
```bash
python examples/performance_test.py
```

## Dependencies 📦

Core dependencies:
- `numpy`: Numerical computations
- `scipy`: Signal processing
- `librosa`: Audio analysis
- `praat-parselmouth`: Voice analysis
- `matplotlib`: Plotting
- `pandas`: Data handling
- `jinja2`: Report templating

See `requirements.txt` for complete list.

## Limitations and Considerations ⚠️

### Technical Limitations
- Requires minimum 1-2 seconds of audio
- Optimized for sustained vowels and connected speech
- Performance depends on audio quality
- Some measures require voiced segments

### Clinical Considerations
- Automated analysis supplements but doesn't replace clinical judgment
- Requires interpretation by qualified professionals
- Cultural and linguistic factors may affect results
- Individual variation in normal ranges

### Validation Status
- Research-grade implementation
- Ongoing clinical validation studies
- Peer review and validation encouraged

## Contributing 🤝

We welcome contributions! Please see:
- Code style guidelines in `CONTRIBUTING.md`
- Issue templates for bug reports
- Pull request process
- Research collaboration opportunities

## License 📋

[Specify your license here - e.g., MIT, GPL-3.0, Apache-2.0]

## Citation 📖

If you use Voice Lab GPT in research, please cite:

```bibtex
@software{voice_lab_gpt,
  title={Voice Lab GPT: Professional Voice and Speech Analysis System},
  author={[Your Name/Organization]},
  year={2024},
  url={https://github.com/your-org/voice-lab-gpt}
}
```

## Support 💬

- **Documentation**: [Link to docs]
- **Issues**: GitHub Issues tracker
- **Discussions**: GitHub Discussions
- **Email**: [support email]

## Acknowledgments 🙏

Voice Lab GPT builds upon decades of voice research and clinical expertise. We acknowledge:
- Voice science research community
- Clinical voice professionals
- Open source audio analysis tools
- International voice analysis standards

---

**🤖 Generated with Voice Lab GPT**

*Professional voice analysis for clinical excellence*