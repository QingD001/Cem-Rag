"""
Utility functions for CEM-RAG project
"""

import logging
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def select_best_gpu() -> str:
    """
    Select GPU with most free memory.
    Uses nvidia-smi to get accurate memory usage across all processes.
    
    Returns:
        Device string (e.g., "cuda:0", "cuda:1") or "cpu" if no GPU available
    """
    if not torch.cuda.is_available():
        return "cpu"
    
    # Try to use nvidia-smi for more accurate memory detection
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            best_gpu = 0
            max_free = 0
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    gpu_idx, free_mem = line.split(', ')
                    gpu_idx = int(gpu_idx)
                    free_mem = int(free_mem)  # MB
                    if free_mem > max_free:
                        max_free = free_mem
                        best_gpu = gpu_idx
            
            logger.info(f"Selected GPU {best_gpu} (free memory: {max_free / 1024:.2f} GB via nvidia-smi)")
            return f"cuda:{best_gpu}"
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        # Fallback to PyTorch method if nvidia-smi is unavailable
        pass
    
    # Fallback: use PyTorch's memory info
    best_gpu = 0
    max_free = 0
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory
        torch.cuda.set_device(i)
        reserved = torch.cuda.memory_reserved(i)
        free_mem = total_mem - reserved
        
        if free_mem > max_free:
            max_free = free_mem
            best_gpu = i
    
    logger.info(f"Selected GPU {best_gpu} (free memory: {max_free / 1024**3:.2f} GB)")
    return f"cuda:{best_gpu}"


def resolve_huggingface_model_path(
    base_path: Path,
    model_name_for_hub: Optional[str] = None,
) -> Path:
    """
    Resolve the actual model path from HuggingFace cache structure.
    
    HuggingFace cache structure:
        base_path/
          models--{org}--{model-name}/
            snapshots/
              {commit_hash}/  ← actual model files here
            refs/
            blobs/
    
    Args:
        base_path: Base directory containing HuggingFace cache (e.g., project_root / "models")
        model_name_for_hub: Optional HuggingFace model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
                          If provided, will look for specific model directory.
                          If None, will search for any models--* directory.
    
    Returns:
        Path to the actual model directory (snapshot directory with config.json)
    
    Raises:
        ValueError: If model path cannot be resolved or config.json not found
    """
    if not base_path.exists():
        raise ValueError(f"Base path does not exist: {base_path}")
    
    model_path = base_path
    
    # If we know the model name, look for the specific hub directory
    if model_name_for_hub:
        # Convert "Qwen/Qwen3-Embedding-0.6B" to "models--Qwen--Qwen3-Embedding-0.6B"
        hub_dir_name = f"models--{model_name_for_hub.replace('/', '--')}"
        hub_dir = base_path / hub_dir_name
        if hub_dir.exists():
            snapshots_dir = hub_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    model_path = snapshots[0]
                    logger.info(f"Found HuggingFace cache structure, using snapshot: {model_path}")
    
    # Fallback: search for any models--* directory
    if model_path == base_path:
        hub_dirs = list(base_path.glob("models--*"))
        if hub_dirs:
            hub_dir = hub_dirs[0]
            snapshots_dir = hub_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    model_path = snapshots[0]
                    logger.info(f"Found HuggingFace cache structure, using snapshot: {model_path}")
        elif (base_path / "snapshots").exists():
            # Direct snapshots directory (unusual but possible)
            snapshots = list((base_path / "snapshots").iterdir())
            if snapshots:
                model_path = snapshots[0]
                logger.info(f"Found snapshot directory: {model_path}")
    
    # Verify config.json exists
    if not (model_path / "config.json").exists():
        raise ValueError(f"config.json not found in model path: {model_path}")
    
    return model_path

