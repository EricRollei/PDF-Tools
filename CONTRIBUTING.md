# Contributing to PDF Tools for ComfyUI

Thank you for your interest in contributing to PDF Tools! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by common sense and mutual respect. We expect all contributors to:

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

1. **Clear title and description**
2. **Steps to reproduce** the problem
3. **Expected behavior** vs **actual behavior**
4. **Environment details** (OS, Python version, ComfyUI version)
5. **Relevant logs or error messages**
6. **Screenshots** if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:

1. **Clear description** of the feature
2. **Use case** - why would this be useful?
3. **Examples** of how it would work
4. **References** to similar features in other tools (if applicable)

### Contributing Code

1. **Fork the repository** and create a branch from `main`
2. **Make your changes** following our coding standards
3. **Test thoroughly** with real data
4. **Update documentation** if needed
5. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.10 or higher
- ComfyUI installation
- Git

### Setting Up Development Environment

```powershell
# Clone your fork
git clone https://github.com/YOUR_USERNAME/PDF_tools.git
cd PDF_tools

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if any)
pip install pytest black flake8
```

### Testing Your Changes

1. **Copy to ComfyUI custom_nodes:**
   ```powershell
   # Create symlink or copy to ComfyUI
   # Example: mklink /D "C:\ComfyUI\custom_nodes\PDF_tools" "C:\path\to\your\PDF_tools"
   ```

2. **Start ComfyUI and test:**
   - Verify nodes load correctly
   - Test with real-world data
   - Check for errors in console

3. **Run manual tests:**
   - Test each modified node
   - Verify backward compatibility
   - Check error handling

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (prefer 80-90)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Group in order (standard library, third-party, local)
- **Docstrings**: Use for all public functions, classes, and modules

### File Headers

All Python files should include a header with license and credits:

```python
"""
[Module Name] - [Brief Description]

Author: [Your Name] (GitHub: YourUsername)
Original Author: Eric Hiss (GitHub: EricRollei)
Contact: [eric@historic.camera, eric@rollei.us]
License: Dual License (CC BY-NC 4.0 / Commercial)
Copyright (c) 2025 Eric Hiss. All rights reserved.

[Additional description if needed]

Dependencies:
- List key dependencies used by this specific module
- Include their licenses

[Additional notes]
"""
```

### Code Organization

```python
# 1. Standard library imports
import os
import sys
from typing import Dict, List, Optional

# 2. Third-party imports
import numpy as np
import torch
from PIL import Image

# 3. Local imports
from .utils import helper_function
```

### Error Handling

Always include comprehensive error handling:

```python
try:
    # Your code here
    result = process_image(image)
except ValueError as e:
    print(f"Error: Invalid input - {e}")
    return None
except Exception as e:
    print(f"Unexpected error in process_image: {e}")
    import traceback
    traceback.print_exc()
    return None
```

### Logging and Debug Output

Use print statements for user-facing messages:

```python
print(f"[PDF Extractor] Processing {filename}")
print(f"[PDF Extractor] Extracted {count} images")
print(f"[PDF Extractor] ERROR: {error_message}")
```

### ComfyUI Node Structure

Follow the established pattern for ComfyUI nodes:

```python
class MyCustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "param1": ("STRING", {"default": ""}),
            },
            "optional": {
                "param2": ("INT", {"default": 0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "PDF_tools"
    
    def process(self, param1, param2=0):
        """Process the input and return result."""
        try:
            # Your logic here
            return (result,)
        except Exception as e:
            print(f"Error in MyCustomNode: {e}")
            return (None,)
```

## Testing

### Manual Testing

1. **Basic functionality**: Does the node work as expected?
2. **Edge cases**: Empty inputs, invalid data, very large files
3. **Error handling**: Graceful failures with helpful messages
4. **Performance**: Reasonable speed for typical use cases
5. **Memory usage**: No memory leaks or excessive consumption

### Test Data

- Use realistic test data (actual PDFs, images, URLs)
- Test with both small and large inputs
- Test with corrupted or malformed data
- Document any test cases in `Docs/` if they're useful examples

### Regression Testing

- Ensure existing functionality still works
- Test with existing workflows if possible
- Check for backward compatibility

## Submitting Changes

### Pull Request Process

1. **Update documentation** for any user-facing changes
2. **Add or update tests** if applicable
3. **Ensure your code follows** the coding standards
4. **Write a clear PR description** including:
   - What changes were made
   - Why they were needed
   - How to test them
   - Any breaking changes

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How was this tested?

## Screenshots (if applicable)
[Add screenshots here]

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Tested with real data
- [ ] No breaking changes (or documented if necessary)
```

### Commit Messages

Use clear, descriptive commit messages:

```
Good examples:
- "Add support for multi-page PDF processing"
- "Fix memory leak in image processor"
- "Update README with new installation instructions"

Avoid:
- "fix bug"
- "update"
- "changes"
```

## License

By contributing to this project, you agree that your contributions will be licensed under the same dual license as the project (CC BY-NC 4.0 for non-commercial use, separate commercial license required).

You also certify that:

1. You have the right to contribute the code
2. Your contribution is your original work or properly attributed
3. You understand the licensing terms

## Questions?

If you have questions about contributing:

- **Open an issue** for general questions
- **Email**: eric@historic.camera or eric@rollei.us
- **Check existing issues** for similar questions

## Acknowledgments

Thank you for taking the time to contribute! Your efforts help make this project better for everyone.

---

**Remember**: This is a working codebase used in production. Please test thoroughly and maintain backward compatibility whenever possible.
