# Contributing to Voice Lab GPT

We welcome contributions to Voice Lab GPT! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Types of Contributions
- 🐛 **Bug Reports**: Report issues or unexpected behavior
- ✨ **Feature Requests**: Suggest new features or improvements
- 📖 **Documentation**: Improve documentation and examples
- 🧪 **Testing**: Add tests or improve test coverage
- 🔧 **Code**: Fix bugs or implement new features

### Getting Started

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/voice-lab-gpt.git
   cd voice-lab-gpt
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -e .
   
   # Run tests to ensure everything works
   python tests/test_voice_lab.py
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

### Development Guidelines

#### Code Style
- Follow PEP 8 Python style guidelines
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

#### Testing
- Write tests for new features
- Ensure all tests pass before submitting
- Aim for good test coverage
- Include both unit tests and integration tests

#### Documentation
- Update README.md if needed
- Add docstrings to new functions
- Update examples if adding new features
- Include usage examples in docstrings

### Submitting Changes

1. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

2. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Go to GitHub and create a pull request
   - Provide clear description of changes
   - Reference any related issues
   - Include screenshots if relevant

### Pull Request Guidelines

#### PR Description Should Include:
- **What**: Brief description of changes
- **Why**: Reason for the changes
- **How**: Technical details if complex
- **Testing**: How you tested the changes
- **Screenshots**: If UI changes are involved

#### PR Checklist:
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated if needed
- [ ] No breaking changes (or clearly documented)

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python tests/test_voice_lab.py

# Run specific test
python -m pytest tests/test_voice_lab.py::TestVoiceLabGPT::test_analyze_audio_array
```

### Adding Tests
- Place tests in the `tests/` directory
- Follow existing test patterns
- Test both success and failure cases
- Include edge cases and error conditions

## 📋 Issue Guidelines

### Bug Reports
Please include:
- **Environment**: OS, Python version, package versions
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Audio Sample**: If relevant (anonymized if needed)
- **Error Messages**: Full error messages and stack traces

### Feature Requests
Please include:
- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Any alternative approaches considered?
- **Clinical Context**: How it relates to voice analysis workflows

## 🏥 Clinical Considerations

### When Contributing Clinical Features:
- Ensure clinical accuracy and evidence-based approaches
- Consider international standards and practices
- Validate against established voice analysis methods
- Include appropriate disclaimers and limitations
- Consider ethical implications of voice analysis

### Voice Analysis Standards:
- Follow established acoustic analysis protocols
- Use validated measures and calculations
- Provide references for clinical interpretations
- Consider cross-cultural and linguistic factors

## 📚 Resources

### Voice Analysis References:
- Titze, I. R. (1994). Principles of voice production
- Baken, R. J., & Orlikoff, R. F. (2000). Clinical measurement of speech and voice
- Maryn, Y., et al. (2009). The acoustic voice quality index
- International standards for voice analysis

### Development Resources:
- [Python Style Guide (PEP 8)](https://pep8.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [SciPy Documentation](https://scipy.org/doc/)
- [Librosa Documentation](https://librosa.org/doc/)

## 💬 Community

### Communication Channels:
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code review and collaboration

### Code of Conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and improve
- Consider diverse perspectives and use cases
- Maintain professionalism in all interactions

## 🏷️ Release Process

### Version Numbering:
We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist:
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version numbers updated
- [ ] Changelog updated
- [ ] Performance regression testing
- [ ] Clinical validation if needed

## ✨ Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Academic publications if applicable

## 🔒 Security

For security-related issues:
- Do not open public issues
- Contact maintainers directly
- Provide detailed information privately
- Allow time for responsible disclosure

---

Thank you for contributing to Voice Lab GPT! Your contributions help advance professional voice analysis and support clinicians and researchers worldwide. 🎙️

**Questions?** Feel free to open an issue or start a discussion on GitHub.