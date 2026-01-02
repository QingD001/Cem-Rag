#!/bin/bash
# Setup script for evaluation environment

set -e

echo "Setting up evaluation environment for CEM-RAG..."

# Check Python version
python3 --version

# Install core dependencies
echo "Installing core dependencies..."
pip3 install --upgrade pip
pip3 install torch transformers numpy tqdm

# Install mteb (try to get LongEmbed support)
echo "Installing mteb..."
pip3 install mteb

# Try to install from source if LongEmbed is not available
echo "Checking for LongEmbed support..."
python3 -c "from mteb.tasks import LongEmbedRetrieval" 2>/dev/null || {
    echo "LongEmbed not found in installed mteb. Installing from source..."
    pip3 install git+https://github.com/embeddings-benchmark/mteb.git
}

echo "Setup complete!"
echo ""
echo ""
echo "To install the package in development mode:"
echo "  pip install -e ."
echo ""
echo "To verify installation, run:"
echo "  python3 -c 'from mteb import MTEB; print(\"MTEB installed successfully\")'"
echo ""
echo "To check LongEmbed support:"
echo "  python3 -c 'from mteb.tasks import LongEmbedRetrieval; print(\"LongEmbed available\")'"
echo ""
echo "To run evaluation:"
echo "  python3 src/eval/longembed_eval.py --model Qwen/Qwen3-Embedding-0.6B"

