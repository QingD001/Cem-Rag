#!/usr/bin/env python3
"""
Script to download and verify embedding model from HuggingFace

Usage:
    # Activate virtual environment first
    source venv/bin/activate
    
    # Set HuggingFace token (optional but recommended for rate limits)
    export HF_TOKEN=your_token_here
    # Get token from: https://huggingface.co/settings/tokens
    
    # Then run the script
    python scripts/download_model.py Qwen/Qwen3-Embedding-0.6B

Requirements:
    - transformers >= 4.50.0 (for Qwen3 support)
    - torch
    - numpy < 2.0
    
To set up the environment:
    python3 -m virtualenv venv
    source venv/bin/activate
    pip install torch transformers 'numpy<2' tqdm
"""

import os
import sys
from pathlib import Path

# Clear any mirror endpoint to use official HuggingFace
if 'HF_ENDPOINT' in os.environ and 'hf-mirror' in os.environ.get('HF_ENDPOINT', ''):
    del os.environ['HF_ENDPOINT']

# Enable progress bars and ensure output is flushed
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'

# Ensure output is flushed for real-time progress
def print_flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)

import torch
from transformers import AutoModel, AutoTokenizer

def get_model_dir(model_name: str, base_dir: str = "models") -> Path:
    """Get the directory path for storing the model"""
    # Simply use models/ as cache_dir, HuggingFace will create its own structure:
    # models/models--Qwen--Qwen3-Embedding-0.6B/
    # This avoids unnecessary nesting
    project_root = Path(__file__).parent.parent
    model_dir = project_root / base_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def download_model(model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
    """Download and verify the embedding model"""
    
    print_flush(f"Downloading model: {model_name}")
    print_flush("=" * 80)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_flush(f"Using device: {device}")
    
    # Download and load model
    print_flush(f"\nDownloading model from HuggingFace...")
    print_flush("This may take a while depending on your network speed...")
    
    # Get model directory
    model_dir = get_model_dir(model_name)
    print_flush(f"Model will be saved to: {model_dir}")
    
    # Check for HuggingFace token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print_flush("✓ Using HuggingFace token for authentication")
    else:
        print_flush("ℹ️  No HF_TOKEN found. You can set it with: export HF_TOKEN=your_token")
        print_flush("   Get your token from: https://huggingface.co/settings/tokens")
    
    try:
        print_flush("\n[1/3] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=str(model_dir),
            token=hf_token
        )
        print_flush("✓ Tokenizer downloaded")
        
        print_flush("\n[2/3] Downloading model (this is the large file, please wait)...")
        print_flush("   Progress will be shown below:")
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=str(model_dir),
            token=hf_token
        )
        print_flush("✓ Model downloaded")
        
        print_flush("\n[3/3] Moving model to device...")
        model.to(device)
        model.eval()
        print_flush(f"✓ Model loaded successfully")
    except Exception as e:
        error_msg = str(e)
        print_flush(f"✗ Failed to load model: {error_msg}")
        
        # Check if it's a version issue
        if "does not recognize this architecture" in error_msg or "model type" in error_msg.lower():
            print_flush(f"\n⚠️  This model requires a newer version of transformers.")
            print_flush(f"   Current version: {__import__('transformers').__version__}")
            print_flush(f"   Try: pip install --upgrade transformers")
            print_flush(f"   Or install from source: pip install git+https://github.com/huggingface/transformers.git")
        
        return False
    
    # Check model info
    print_flush(f"\nModel Information:")
    print_flush(f"  Hidden size: {model.config.hidden_size}")
    print_flush(f"  Max position embeddings: {getattr(model.config, 'max_position_embeddings', 'N/A')}")
    print_flush(f"  Model type: {model.config.model_type}")
    
    # Test encoding
    print_flush(f"\nTesting encoding...")
    test_texts = [
        "Hello, world!",
        "This is a test sentence.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    try:
        with torch.no_grad():
            inputs = tokenizer(
                test_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_embeddings = sum_embeddings / sum_mask
            
            # Normalize
            normalized = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1)
            
            print_flush(f"✓ Encoding successful")
            print_flush(f"  Input texts: {len(test_texts)}")
            print_flush(f"  Output shape: {normalized.shape}")
            print_flush(f"  Embedding dimension: {normalized.shape[1]}")
            print_flush(f"  Sample embedding (first 5 dims): {normalized[0, :5].cpu().tolist()}")
            
    except Exception as e:
        print_flush(f"✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print_flush("\n" + "=" * 80)
    print_flush("✓ All tests passed!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and verify HuggingFace embedding model")
    parser.add_argument("model", nargs="?", default="Qwen/Qwen3-Embedding-0.6B", 
                       help="Model name or path (default: Qwen/Qwen3-Embedding-0.6B)")
    args = parser.parse_args()
    
    success = download_model(args.model)
    sys.exit(0 if success else 1)

