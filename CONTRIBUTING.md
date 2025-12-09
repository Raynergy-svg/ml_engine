# Contributing to ML Engine

Thank you for your interest in contributing to the ML Engine project! This document provides guidelines and best practices for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help create a welcoming environment for all contributors

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/your-username/ml_engine.git
cd ml_engine
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest black ruff mypy
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

## Development Guidelines

### Code Style

We follow Python best practices:

1. **PEP 8**: Follow PEP 8 style guide
2. **Type Hints**: Add type hints to function signatures
3. **Docstrings**: Use Google-style docstrings
4. **Comments**: Write clear, concise comments for complex logic

Example:
```python
def train_model(
    data: np.ndarray,
    labels: np.ndarray,
    epochs: int = 100
) -> Dict[str, float]:
    """
    Train a machine learning model.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Training labels of shape (n_samples,)
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If data and labels have mismatched shapes
    """
    if len(data) != len(labels):
        raise ValueError("Data and labels must have the same length")
    
    # Implementation here
    return {"loss": 0.5, "accuracy": 0.95}
```

### Testing

1. **Write Tests**: Add tests for new features
2. **Run Tests**: Ensure all tests pass before submitting
3. **Test Coverage**: Aim for >80% code coverage

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Documentation

1. **Update README**: Update README.md for user-facing changes
2. **Inline Documentation**: Add docstrings to all public functions/classes
3. **Examples**: Include usage examples in docstrings
4. **Configuration**: Document new configuration options in config.yaml

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add support for Transformer architecture
fix: Resolve memory leak in data loader
docs: Update installation instructions
refactor: Simplify configuration validation
test: Add unit tests for feature engineering
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code restructuring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No unrelated changes included

### 2. Submit Pull Request

1. Push your branch to GitHub
2. Create a Pull Request from your branch
3. Fill out the PR template
4. Link any related issues

### 3. Code Review

- Respond to review feedback promptly
- Make requested changes in new commits
- Update PR description if scope changes

### 4. After Merge

- Delete your feature branch
- Pull the latest main branch
- Start a new branch for the next feature

## Project Structure

```
ml_engine/
├── ml_engine.py           # Core ML engine
├── models_enhanced.py     # Model architectures
├── data_loader.py         # Data loading
├── evaluation.py          # Metrics and evaluation
├── config_validator.py    # Configuration validation
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Areas for Contribution

### High Priority

1. **Testing**: Expand test coverage
2. **Documentation**: Improve inline documentation
3. **Performance**: Optimize data loading and training
4. **Bug Fixes**: Address open issues

### Medium Priority

1. **Features**: Add new model architectures
2. **Visualization**: Enhance plotting capabilities
3. **Configuration**: Add more configuration options
4. **Examples**: Create usage examples

### Low Priority

1. **Code Cleanup**: Refactor legacy code
2. **Typing**: Add type hints to untyped functions
3. **Logging**: Improve logging messages

## Getting Help

- **Issues**: Check existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Read README.md and IMPROVEMENTS.md

## Security

If you discover a security vulnerability:
1. **Do not** open a public issue
2. Email the maintainers directly
3. Provide detailed information about the vulnerability

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:
- Listed in the project README
- Mentioned in release notes
- Credited in commits

Thank you for contributing to ML Engine! 🚀
