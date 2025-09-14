# Contributing to RedactAI

Thank you for your interest in contributing to RedactAI! This document provides guidelines and information for contributors.

## 🤝 **How to Contribute**

### **Reporting Issues**
- Use the GitHub issue tracker to report bugs or request features
- Provide detailed information about the issue including:
  - Steps to reproduce
  - Expected vs actual behavior
  - System information (OS, Python version, etc.)
  - Error messages or logs

### **Submitting Pull Requests**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Update documentation as needed
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📋 **Development Guidelines**

### **Code Style**
- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all functions and classes
- Keep functions small and focused on a single responsibility

### **Testing**
- Write unit tests for all new functionality
- Ensure test coverage remains above 95%
- Run tests locally before submitting PRs
- Use descriptive test names that explain what is being tested

### **Documentation**
- Update relevant documentation files
- Add docstrings to new functions and classes
- Update API documentation if applicable
- Include examples in docstrings where helpful

## 🏗️ **Project Structure**

```
RedactAI/
├── core/                    # Advanced core systems
├── modules/                 # Core AI/ML modules
├── security/               # Enterprise security
├── dashboard/              # Advanced analytics
├── api/                    # FastAPI REST endpoints
├── utils/                  # Advanced utilities
├── config/                 # Configuration management
├── tests/                  # Comprehensive test suite
└── docs/                   # Documentation
```

## 🧪 **Testing**

### **Running Tests**
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=modules --cov-report=html

# Run specific test modules
pytest tests/test_integration.py -v
```

### **Test Categories**
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **API Tests**: Test REST API endpoints
- **Performance Tests**: Test system performance

## 🔧 **Development Setup**

### **Prerequisites**
- Python 3.10+
- Git
- Docker (optional)

### **Setup Instructions**
```bash
# Clone the repository
git clone https://github.com/jagan-yetukrui/RedactAI.git
cd RedactAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

## 📝 **Commit Guidelines**

### **Commit Message Format**
```
type(scope): description

[optional body]

[optional footer]
```

### **Types**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

### **Examples**
```
feat(api): add batch processing endpoint
fix(face_detection): resolve memory leak in Haar cascades
docs(readme): update installation instructions
test(integration): add end-to-end processing tests
```

## 🚀 **Release Process**

### **Version Numbering**
We use [Semantic Versioning](https://semver.org/):
- `MAJOR`: Incompatible API changes
- `MINOR`: New functionality in a backwards compatible manner
- `PATCH`: Backwards compatible bug fixes

### **Release Checklist**
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Version number is updated
- [ ] CHANGELOG.md is updated
- [ ] Release notes are prepared

## 🐛 **Bug Reports**

When reporting bugs, please include:

1. **Environment Information**
   - Operating System
   - Python version
   - RedactAI version
   - Dependencies versions

2. **Steps to Reproduce**
   - Clear, numbered steps
   - Sample input files (if applicable)
   - Expected vs actual behavior

3. **Error Information**
   - Full error messages
   - Stack traces
   - Log files

4. **Additional Context**
   - Screenshots (if applicable)
   - Related issues
   - Workarounds (if any)

## 💡 **Feature Requests**

When requesting features, please include:

1. **Use Case**
   - Why is this feature needed?
   - How would it be used?

2. **Proposed Solution**
   - Detailed description of the feature
   - API design (if applicable)
   - UI mockups (if applicable)

3. **Alternatives**
   - Other solutions considered
   - Workarounds currently used

## 📞 **Getting Help**

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For general questions and discussions
- **Email**: Contact the maintainers directly

## 🙏 **Recognition**

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to RedactAI! 🎉
