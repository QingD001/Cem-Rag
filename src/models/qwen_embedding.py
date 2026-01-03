"""
Qwen Embedding Model Wrapper for MTEB Evaluation

This module provides a simple wrapper for Qwen3 embedding models.
Supports versions: 0.6B, 4B, 8B
"""

import logging
from pathlib import Path
from typing import List, Optional

from src.models.base_embedding import BaseEmbeddingModel
from src.utils import resolve_huggingface_model_path

logger = logging.getLogger(__name__)

# Model name mapping: version -> HuggingFace model name
# All models are stored in models/ directory, HuggingFace will create:
# models/models--Qwen--Qwen3-Embedding-XXX/
MODEL_NAMES = {
    "0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "4b": "Qwen/Qwen3-Embedding-4B",
    "8b": "Qwen/Qwen3-Embedding-8B",
}


class QwenEmbeddingModel(BaseEmbeddingModel):
    """Qwen3 Embedding Model wrapper for MTEB"""
    
    def __init__(self, version: str = "0.6b", device: str = "cuda", project_root: Optional[Path] = None):
        """
        Initialize Qwen embedding model.
        
        Args:
            version: Model version ("0.6b", "4b", "8b") or full path to model directory
            device: Device to run model on ('cuda', 'cpu', or 'cuda:0', 'cuda:1', etc.)
            project_root: Project root directory (default: auto-detect from file location)
        """
        # Determine project root
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        # Determine model path and model name for HuggingFace
        model_name_for_hub = None
        if version.lower() in MODEL_NAMES:
            model_name_for_hub = MODEL_NAMES[version.lower()]
            base_path = project_root / "models"
        else:
            # Assume it's a full path (relative or absolute)
            base_path = Path(version)
            if not base_path.is_absolute():
                base_path = project_root / base_path
        
        # Resolve HuggingFace cache structure to actual model path
        model_path = resolve_huggingface_model_path(base_path, model_name_for_hub)
        
        # Determine model name for MTEB metadata
        if version.lower() in MODEL_NAMES:
            model_name = MODEL_NAMES[version.lower()]
        else:
            # For custom paths, construct a name (fallback)
            model_name = f"custom/{Path(version).name}" if Path(version).is_absolute() else f"custom/{version}"
        
        self.version = version
        
        # Call parent class __init__
        super().__init__(
            model_name_for_hub=model_name,
            model_path=model_path,
            device=device,
            project_root=project_root,
        )
    

