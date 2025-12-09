# ML Engine Examples

This directory contains example scripts demonstrating how to use the ML Engine.

## Available Examples

### basic_usage.py

A simple example showing the core workflow:
- Loading configuration
- Initializing the ML engine
- Training a model with dummy data
- Evaluating model performance

**Run it:**
```bash
python basic_usage.py
```

## Creating Your Own Examples

When creating new examples:
1. Add the parent directory to Python path
2. Use descriptive variable names
3. Add comments explaining each step
4. Include error handling
5. Print progress and results

Example template:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_engine import EnhancedMLEngine

def main():
    # Your example code here
    pass

if __name__ == "__main__":
    main()
```
