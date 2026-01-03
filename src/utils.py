"""
Utility functions for CEM-RAG project
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

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


def get_gpu_memory_info(device: str = None) -> dict:
    """
    Get GPU memory usage information.
    
    Args:
        device: Device string (e.g., "cuda:0"). If None, uses current device.
    
    Returns:
        Dictionary with memory info: {
            'allocated_gb': float,
            'reserved_gb': float,
            'max_allocated_gb': float,
            'total_gb': float,
            'free_gb': float
        }
        Returns None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None
    
    if device:
        torch.cuda.set_device(device)
    
    device_id = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)
    max_allocated = torch.cuda.max_memory_allocated(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    
    return {
        'allocated_gb': allocated / 1024**3,
        'reserved_gb': reserved / 1024**3,
        'max_allocated_gb': max_allocated / 1024**3,
        'total_gb': total / 1024**3,
        'free_gb': (total - reserved) / 1024**3,
    }


def log_gpu_memory(device: str = None, label: str = "GPU memory"):
    """
    Log GPU memory usage information.
    
    Args:
        device: Device string (e.g., "cuda:0"). If None, uses current device.
        label: Label for the log message.
    """
    info = get_gpu_memory_info(device)
    if info is None:
        logger.info(f"{label}: CUDA not available")
        return
    
    logger.info(
        f"{label}: allocated={info['allocated_gb']:.2f} GB, "
        f"reserved={info['reserved_gb']:.2f} GB, "
        f"max_allocated={info['max_allocated_gb']:.2f} GB, "
        f"free={info['free_gb']:.2f} GB / {info['total_gb']:.2f} GB"
    )


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


def _move_item_with_overwrite(item: Path, target: Path, source_name: str = ""):
    """Move item to target, overwriting if target exists."""
    if target.exists():
        if target.is_file() and item.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    
    if item.is_file():
        item.rename(target)
    elif item.is_dir():
        shutil.move(str(item), str(target))


def _flatten_revision_subdirectory(revision_dir: Path, target_dir: Path):
    """Flatten a revision subdirectory by moving all items to target directory."""
    moved_any = False
    for item in revision_dir.iterdir():
        target = target_dir / item.name
        _move_item_with_overwrite(item, target, revision_dir.name)
        moved_any = True
    
    if moved_any:
        try:
            revision_dir.rmdir()
            logger.debug(f"Flattened directory: removed {revision_dir} and moved files to {target_dir}")
        except OSError:
            pass


def _merge_directories(source_dir: Path, target_dir: Path):
    """Merge contents from source_dir to target_dir, overwriting conflicts."""
    for item in list(source_dir.iterdir()):
        target_item = target_dir / item.name
        _move_item_with_overwrite(item, target_item, source_dir.name)
    
    try:
        source_dir.rmdir()
        logger.debug(f"Merged and removed directory: {source_dir}")
    except OSError:
        logger.warning(f"Could not remove directory {source_dir} after merging (not empty)")


def flatten_results_directory(output_dir: str, model_name: str):
    """
    Flatten MTEB results directory structure.
    
    When using custom ResultCache with cache_path=output_dir, MTEB saves results to:
    output_dir/results/{model_name}/{revision}/{task_name}.json
    
    This function flattens the structure to:
    output_dir/{model_name}/{task_name}.json
    
    Structure before: output_dir/results/Qwen__Qwen3-Embedding-0.6B/no_revision_available/file.json
    Structure after:  output_dir/Qwen3-Embedding-0.6B/file.json
    
    Args:
        output_dir: Output directory containing MTEB results
        model_name: Model name in MTEB format (e.g., "Qwen/Qwen3-Embedding-0.6B").
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    # Convert model name to MTEB directory format (e.g., "Qwen/Qwen3-Embedding-0.6B" -> "Qwen__Qwen3-Embedding-0.6B")
    mteb_dir_name = model_name.replace('/', '__')
    
    # With custom ResultCache, MTEB saves to: output_dir/results/{model_name}/{revision}/
    results_path = output_path / "results"
    model_dir = results_path / mteb_dir_name if results_path.exists() else None
    
    if not model_dir or not model_dir.exists():
        return
    
    # Process the model directory
    original_name = model_dir.name
    
    # First, flatten revision subdirectories in the current directory
    # MTEB typically creates revision subdirectories like 'no_revision_available'
    for revision_dir in list(model_dir.iterdir()):
        if revision_dir.is_dir():
            _flatten_revision_subdirectory(revision_dir, model_dir)
    
    # Remove organization prefix from directory name
    # (e.g., "Qwen__Qwen3-Embedding-0.6B" -> "Qwen3-Embedding-0.6B")
    parts = original_name.split('__')
    if len(parts) < 2:
        return
    
    new_name = '__'.join(parts[1:])
    if not new_name or new_name == original_name:
        return
    
    # Move from output_dir/results/{model_name} to output_dir/{new_name}
    new_model_dir = output_path / new_name
    
    if new_model_dir.exists():
        # Target exists: flatten it first, then merge
        for target_revision_dir in list(new_model_dir.iterdir()):
            if target_revision_dir.is_dir():
                _flatten_revision_subdirectory(target_revision_dir, new_model_dir)
        _merge_directories(model_dir, new_model_dir)
        logger.debug(f"Merged directory: {original_name} -> {new_name}")
    else:
        # Move from results/ subdirectory to output_dir root
        import shutil
        shutil.move(str(model_dir), str(new_model_dir))
        logger.debug(f"Moved directory from results/: {original_name} -> {new_name}")



def sort_top_level_keys(data: Dict) -> Dict:
    """Sort only the top-level keys of a dictionary, preserving nested structure."""
    return {key: data[key] for key in sorted(data.keys())}


def select_device(device: str) -> str:
    """
    Select and validate device for model loading.
    
    Args:
        device: Device string ('cpu', 'cuda', or 'cuda:0', 'cuda:1', etc.)
        
    Returns:
        Validated device string
    """
    if device == "cpu":
        return "cpu"
    elif device.startswith("cuda"):
        if torch.cuda.is_available():
            if ":" in device:
                return device  # e.g., "cuda:1"
            else:
                # Auto-select GPU with most free memory
                return select_best_gpu()
        else:
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
    else:
        return "cpu"


def get_effective_batch_size(max_length: int, batch_size: int, batch_texts_length: int) -> int:
    """
    Calculate effective batch size based on max_length to avoid OOM.
    
    For long sequences, reduce batch size to prevent out-of-memory errors.
    
    Args:
        max_length: Maximum sequence length supported by the model
        batch_size: Original batch size requested
        batch_texts_length: Number of texts in the current batch
        
    Returns:
        Effective batch size to use for processing
    """
    if max_length > 16384:
        # For extremely long sequences (32768+), use batch_size=1 to avoid OOM
        effective_batch_size = 1
        if batch_texts_length > 1:
            logger.debug(f"Reducing batch size to 1 for max_length={max_length} to avoid OOM")
    elif max_length > 8192:
        # For very long sequences, use small batch size
        effective_batch_size = min(4, batch_texts_length)
        if batch_texts_length > 4:
            logger.debug(f"Reducing batch size to 4 for max_length={max_length} to avoid OOM")
    elif max_length > 2048:
        # For medium sequences, use moderate batch size
        effective_batch_size = min(8, batch_texts_length)
    else:
        # For short sequences, use original batch size
        effective_batch_size = min(batch_size, batch_texts_length)
    
    return effective_batch_size
