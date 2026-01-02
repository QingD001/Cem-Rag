#!/bin/bash
# Setup script for virtual environment

set -e

echo "Setting up virtual environment for CEM-RAG..."

# Check if venv already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Activate it with:"
    echo "  source venv/bin/activate"
    exit 0
fi

# Try to use conda if available, otherwise use venv
if command -v conda &> /dev/null || [ -f ~/anaconda3/bin/conda ]; then
    echo "Using conda to create Python 3.11 environment..."
    CONDA_BIN=$(command -v conda || echo ~/anaconda3/bin/conda)
    $CONDA_BIN create -p ./venv python=3.11 -y
    echo "✓ Conda environment created"
else
    echo "Using venv (requires Python 3.10+)..."
    # Try Python 3.11, 3.10, or 3.9
    if command -v python3.11 &> /dev/null; then
        python3.11 -m venv venv || python3.11 -m virtualenv venv
    elif command -v python3.10 &> /dev/null; then
        python3.10 -m venv venv || python3.10 -m virtualenv venv
    elif command -v python3.9 &> /dev/null; then
        python3.9 -m venv venv || python3.9 -m virtualenv venv
    else
        echo "Error: Need Python 3.9+ to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    # Standard venv
    source venv/bin/activate
elif [ -f "venv/etc/profile.d/conda.sh" ] || [ -d "venv/conda-meta" ]; then
    # Conda environment
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate ./venv
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install torch transformers 'numpy<2' tqdm

echo ""
echo "✓ Virtual environment setup complete!"
echo ""
echo "Python version: $(python --version)"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To download the model, run:"
echo "  python scripts/download_model.py Qwen/Qwen3-Embedding-0.6B"

