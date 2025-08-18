"""
Basic usage examples for Voice Lab GPT
Demonstrates core functionality and typical use cases
"""

import sys
import os
import numpy as np

# Add the parent directory to the path to import voice_lab
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voice_lab import VoiceLabGPT

def example_file_analysis():
    """Example: Analyze an audio file"""
    print("=" * 60)
    print("EXAMPLE 1: File Analysis")
    print("=" * 60)
    
    # Initialize Voice Lab GPT
    voice_lab = VoiceLabGPT(
        sr=16000,
        enable_visualizations=True
    )
    
    # For this example, we'll create a synthetic audio file
    # In real use, you would provide path to actual audio file
    sample_rate = 16000
    duration = 3.0  # 3 seconds
    
    # Generate synthetic vowel /a/ with some perturbations
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 150 + 10 * np.sin(2 * np.pi * 3 * t)  # F0 with slight vibrato
    
    # Add some jitter and shimmer
    jitter = 0.01 * np.random.randn(len(t))
    shimmer = 0.05 * np.random.randn(len(t))
    
    # Generate harmonic series with noise
    audio = np.zeros_like(t)
    for harmonic in range(1, 6):
        frequency = f0 * harmonic * (1 + jitter)
        amplitude = (0.8 / harmonic) * (1 + shimmer)
        audio += amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add some breathiness (noise)
    noise = 0.1 * np.random.randn(len(t))
    audio += noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Analyze the synthetic audio
    results = voice_lab.analyze_audio_array(
        audio,
        voice_type='female',
        patient_info='Example Patient - Synthetic /a/',
        task_description='Sustained vowel /a/ (synthetic)',
        generate_reports=True,
        generate_visualizations=True
    )
    
    # Display key results
    if 'error' not in results:
        print("✅ Analysis completed successfully!")
        print()
        
        print("📊 GRBAS Ratings:")
        grbas = results['grbas_results']
        print(f"  G (Grade):     {grbas['G']}/3 (confidence: {grbas['confidence']['G']:.0%})")
        print(f"  R (Roughness): {grbas['R']}/3 (confidence: {grbas['confidence']['R']:.0%})")
        print(f"  B (Breathiness): {grbas['B']}/3 (confidence: {grbas['confidence']['B']:.0%})")
        print(f"  A (Asthenia):  {grbas['A']}/3 (confidence: {grbas['confidence']['A']:.0%})")
        print(f"  S (Strain):    {grbas['S']}/3 (confidence: {grbas['confidence']['S']:.0%})")
        print(f"  Overall Confidence: {grbas['confidence']['overall']:.0%}")
        print()
        
        print("🔬 Key Acoustic Features:")
        acoustic = results['acoustic_features']
        print(f"  F0 Mean:       {acoustic['f0_mean']:.1f} Hz")
        print(f"  F0 SD:         {acoustic['f0_std']:.1f} Hz")
        print(f"  Jitter:        {acoustic['jitter_local']:.4f} ({acoustic['jitter_local']*1000:.1f}‰)")
        print(f"  Shimmer:       {acoustic['shimmer_local']:.4f} ({acoustic['shimmer_local']*100:.1f}%)")
        print(f"  HNR:           {acoustic['hnr_mean']:.1f} dB")
        print(f"  CPP:           {acoustic['cpp_mean']:.1f} dB")
        print()
        
        print("🏥 Clinical Impression:")
        clinical = results['clinical_results']
        primary = clinical['primary_impression']
        print(f"  Primary:       {primary['condition'].replace('_', ' ').title()}")
        print(f"  Description:   {primary['description']}")
        print(f"  Confidence:    {primary['confidence']:.0%}")
        print()
        
        print("🚨 Triage:")
        triage = clinical['triage']
        print(f"  Level:         {triage['level'].upper()}")
        print(f"  Timeframe:     {triage['timeframe']}")
        print(f"  Specialist:    {triage['specialist']}")
        print()
        
        if clinical.get('red_flags'):
            print("⚠️  Red Flags:")
            for flag in clinical['red_flags']:
                print(f"    • {flag}")
            print()
        
        print("📋 Analysis Quality:")
        quality = results['quality_assessment']
        print(f"  Overall Quality: {quality['overall_quality'].title()}")
        print(f"  Reliability:     {quality['reliability_score']:.0%}")
        if quality['warnings']:
            print("  Warnings:")
            for warning in quality['warnings']:
                print(f"    • {warning}")
        
        print()
        print("📈 Generated Reports:")
        if results.get('reports'):
            reports = results['reports']
            for report_type in reports.keys():
                if report_type != 'plots':
                    print(f"  ✓ {report_type.replace('_', ' ').title()}")
        
        if results.get('visualizations'):
            print("📊 Generated Visualizations:")
            for plot_name in results['visualizations'].keys():
                if plot_name != 'error':
                    print(f"  ✓ {plot_name.replace('_', ' ').title()}")
    
    else:
        print("❌ Analysis failed:")
        print(f"  Error: {results['error']}")
    
    return results

def example_clinical_summary():
    """Example: Get clinical summary"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Clinical Summary")
    print("=" * 60)
    
    voice_lab = VoiceLabGPT()
    
    # Use results from previous analysis if available
    if hasattr(voice_lab, 'last_analysis') and voice_lab.last_analysis:
        summary = voice_lab.get_clinical_summary()
        print("Clinical Summary:")
        print("-" * 40)
        print(summary)
    else:
        print("No previous analysis available for summary.")

def example_grbas_interpretation():
    """Example: GRBAS interpretation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: GRBAS Interpretation")
    print("=" * 60)
    
    # Simulate different GRBAS patterns
    grbas_examples = [
        {"G": 0, "R": 0, "B": 0, "A": 0, "S": 0, "description": "Normal voice"},
        {"G": 2, "R": 0, "B": 3, "A": 1, "S": 0, "description": "Glottic insufficiency"},
        {"G": 3, "R": 3, "B": 1, "A": 0, "S": 1, "description": "Vocal fold lesions"},
        {"G": 2, "R": 1, "B": 1, "A": 2, "S": 3, "description": "Muscle tension dysphonia"}
    ]
    
    print("GRBAS Pattern Examples:")
    print("-" * 40)
    
    for example in grbas_examples:
        grbas_string = f"G{example['G']}R{example['R']}B{example['B']}A{example['A']}S{example['S']}"
        print(f"{grbas_string:8} - {example['description']}")
    
    print()
    print("Interpretation Guide:")
    print("-" * 20)
    print("0 = Normal")
    print("1 = Mild deviation")
    print("2 = Moderate deviation") 
    print("3 = Severe deviation")

def example_comparative_analysis():
    """Example: Compare two analyses (simulated)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Comparative Analysis")
    print("=" * 60)
    
    # This would typically be used to compare pre/post treatment
    # For demo purposes, we'll simulate two different voice conditions
    
    print("This example would typically compare:")
    print("• Pre-treatment vs Post-treatment")
    print("• Different recording conditions")  
    print("• Progress monitoring over time")
    print()
    print("Comparison metrics include:")
    print("• GRBAS rating changes")
    print("• Acoustic measure improvements")
    print("• Clinical impression evolution")
    print("• Statistical significance testing")

def example_quality_assessment():
    """Example: Analysis quality assessment"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Quality Assessment")
    print("=" * 60)
    
    print("Voice Lab GPT performs automatic quality assessment:")
    print()
    print("Audio Quality Factors:")
    print("• Duration (minimum 1-2 seconds recommended)")
    print("• Signal-to-noise ratio")
    print("• Clipping/distortion detection")
    print("• Loudness normalization success")
    print()
    print("Analysis Quality Factors:")
    print("• Voiced segment detection")
    print("• Feature extraction reliability")
    print("• GRBAS estimation confidence")
    print("• Pattern matching strength")
    print()
    print("Quality Ratings:")
    print("• Excellent (>80% reliability)")
    print("• Good (60-80% reliability)")  
    print("• Fair (40-60% reliability)")
    print("• Poor (<40% reliability)")

def main():
    """Run all examples"""
    print("🎙️  VOICE LAB GPT - EXAMPLE DEMONSTRATIONS")
    print("=" * 80)
    print()
    print("This demo shows the key capabilities of Voice Lab GPT:")
    print("• Comprehensive voice analysis")
    print("• GRBAS perceptual estimation") 
    print("• Clinical scenario mapping")
    print("• Automated report generation")
    print("• Quality assessment")
    print()
    
    try:
        # Run the main example
        results = example_file_analysis()
        
        # Run additional examples
        example_clinical_summary()
        example_grbas_interpretation()
        example_comparative_analysis()
        example_quality_assessment()
        
        print("\n" + "=" * 80)
        print("✅ ALL EXAMPLES COMPLETED")
        print("=" * 80)
        print()
        print("Next steps:")
        print("• Try with your own audio files using: voice_lab.analyze_file('path/to/audio.wav')")
        print("• Explore the generated reports and visualizations") 
        print("• Integrate into your clinical workflow")
        print("• Refer to the documentation for advanced features")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        print("\nThis may be due to missing dependencies. Please ensure all required packages are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()