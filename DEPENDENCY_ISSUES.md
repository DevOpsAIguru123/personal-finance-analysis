# Dependency Issues and Solutions

## Overview
This document outlines the dependency compatibility issues encountered when setting up the docling-rag project and the solutions implemented to resolve them.

## Issues Encountered

### 1. PyTorch Platform Compatibility Issue

**Problem:**
- PyTorch versions 2.3.1+ do not have compatible wheels for macOS x86_64 platform
- Error: `Distribution 'torch==2.7.1' can't be installed because it doesn't have a source distribution or wheel for the current platform`
- Supported platforms for newer PyTorch versions: `manylinux_2_28_aarch64`, `manylinux_2_28_x86_64`, `macosx_11_0_arm64`, `win_amd64`
- Missing: `macosx_13_0_x86_64` (Intel Mac support)

**Root Cause:**
- PyTorch 2.3+ dropped support for older macOS Intel architectures
- `docling` package requires PyTorch >= 2.2.2, which creates a dependency conflict

### 2. Python Version Compatibility Issue

**Problem:**
- Initial setup used Python 3.12
- `deepsearch-glm` (a dependency of `docling`) only supports Python 3.9-3.11
- Error: `deepsearch-glm (v0.18.4) only has wheels with the following Python ABI tags: cp39, cp310, cp311`

**Root Cause:**
- Some dependencies in the docling ecosystem haven't been updated for Python 3.12 compatibility

### 3. Dependency Chain Conflicts

**Problem:**
- Complex dependency resolution conflicts between:
  - `docling` → `docling-ibm-models` → `torch>=2.2.2`
  - Platform-specific PyTorch wheel availability
  - Python version constraints

## Solutions Implemented

### Solution 1: Python Version Downgrade
```bash
# Changed from Python 3.12 to 3.11
echo "3.11" > .python-version
```

**Result:** Resolved `deepsearch-glm` compatibility but PyTorch issue persisted.

### Solution 2: CPU-Only PyTorch Installation
```bash
# Install docling with CPU-only PyTorch from specific index
uv pip install docling --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

**Result:** Successfully installed docling with PyTorch 2.2.2 CPU version compatible with macOS x86_64.

### Solution 3: Temporary Dependency Management
- Removed `docling` from `pyproject.toml` dependencies temporarily
- Installed base dependencies with `uv sync`
- Manually installed `docling` with compatible PyTorch version
- Note: This creates a partially managed dependency state

## Current Status

### Working Configuration
- **Python Version:** 3.11.13
- **PyTorch Version:** 2.2.2 (CPU-only)
- **Platform:** macOS x86_64
- **Package Manager:** uv

### Dependencies Successfully Installed
- Core docling packages: `docling==2.42.2`, `docling-core==2.43.1`, `docling-ibm-models==3.9.0`
- LangChain ecosystem: `langchain>=0.3.26`, `langchain-openai>=0.3.28`
- Vector database: `chromadb>=1.0.15`
- Other dependencies: pandas, numpy, openai, etc.

## Recommendations

### Short-term Workarounds
1. **Use Python 3.11** until all dependencies support Python 3.12
2. **Use CPU-only PyTorch** for macOS x86_64 compatibility
3. **Install docling separately** with specific index URLs

### Long-term Solutions
1. **Wait for PyTorch updates** to restore macOS x86_64 support in newer versions
2. **Monitor docling releases** for improved dependency management
3. **Consider containerization** with Docker for consistent environments
4. **Upgrade to Apple Silicon Mac** for better PyTorch support

## Environment Setup Commands

### Recommended Setup Process
```bash
# 1. Set Python version (required for compatibility)
echo "3.11" > .python-version

# 2. Install base dependencies (everything except docling)
uv sync

# 3. Install docling separately with CPU PyTorch (works on macOS x86_64)
uv pip install docling --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

### Why This Approach Works
- **Step 1**: Ensures Python 3.11 compatibility for all docling dependencies
- **Step 2**: Installs all other dependencies from PyPI with correct versions
- **Step 3**: Installs docling with CPU-only PyTorch that supports macOS x86_64

This approach avoids the complex dependency resolution conflicts that occur when trying to use multiple package indexes simultaneously.

### Verification
```bash
# Check Python version
python --version  # Should be 3.11.x

# Check PyTorch version and device
python -c "import torch; print(f'PyTorch: {torch.__version__}, Device: {torch.device(\"cpu\")}')"

# Check docling installation
python -c "import docling; print('Docling installed successfully')"
```

## Alternative Approaches Considered

### 1. Platform-Specific Requirements
- Could use different PyTorch versions based on platform detection
- Complex to maintain across different development environments

### 2. Docker Containerization
- Would provide consistent environment across platforms
- Adds complexity for local development

### 3. Conda Environment
- Conda might handle PyTorch platform compatibility better
- Would require changing from uv to conda for package management

## Future Monitoring

### Dependencies to Watch
- **PyTorch releases** for macOS x86_64 wheel availability
- **docling updates** for improved dependency management
- **deepsearch-glm updates** for Python 3.12 support

### Platform Considerations
- This setup is specific to macOS x86_64
- Different solutions may be needed for:
  - Apple Silicon Macs (ARM64)
  - Linux systems
  - Windows systems

## Notes
- The CPU-only PyTorch installation may impact performance for ML workloads
- GPU acceleration not available with this configuration
- Consider upgrading hardware or using cloud services for GPU-intensive tasks