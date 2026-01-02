"""
Qwen Embedding Model Wrapper for MTEB Evaluation

This module provides a simple wrapper for Qwen3 embedding models.
Supports versions: 0.6B, 4B, 8B
"""

import logging
from pathlib import Path
from typing import List, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils import select_best_gpu, resolve_huggingface_model_path

logger = logging.getLogger(__name__)

# Model name mapping: version -> HuggingFace model name
# All models are stored in models/ directory, HuggingFace will create:
# models/models--Qwen--Qwen3-Embedding-XXX/
MODEL_NAMES = {
    "0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "4b": "Qwen/Qwen3-Embedding-4B",
    "8b": "Qwen/Qwen3-Embedding-8B",
}

# Import AbsEncoder for MTEB compatibility
try:
    from mteb.models.abs_encoder import AbsEncoder
except ImportError:
    # Fallback if AbsEncoder is not available
    AbsEncoder = object


class QwenEmbeddingModel(AbsEncoder):
    """Qwen3 Embedding Model wrapper for MTEB"""
    
    def __init__(self, version: str = "0.6b", device: str = "cuda", project_root: Path = None):
        """
        Initialize Qwen embedding model.
        
        Args:
            version: Model version ("0.6b", "4b", "8b") or full path to model directory
            device: Device to run model on ('cuda', 'cpu', or 'cuda:0', 'cuda:1', etc.)
            project_root: Project root directory (default: auto-detect from file location)
        """
        # Determine project root
        if project_root is None:
            # Assume this file is in src/models/, so go up 2 levels
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
        
        self.model_path = model_path
        self.version = version
        
        # Call parent class __init__ first
        if AbsEncoder != object:
            super().__init__()
        
        # Set MTEB model metadata AFTER calling super().__init__()
        # ModelMeta requires: name in 'org/model' format, and many required fields
        try:
            from mteb.models.model_meta import ModelMeta
            
            # Determine model name from version - must be in 'organization/model_name' format
            if version.lower() in MODEL_NAMES:
                # Use full HuggingFace model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
                model_name = MODEL_NAMES[version.lower()]
            else:
                # For custom paths, construct a name (fallback)
                model_name = f"custom/{Path(version).name}" if Path(version).is_absolute() else f"custom/{version}"
            
            # Try using None for revision - MTEB might skip creating revision subdirectory
            # If None doesn't work, we might need to accept the revision subdirectory
            # Model name already contains version info (e.g., "Qwen/Qwen3-Embedding-0.6B")
            model_revision = None  # Try None to see if MTEB skips revision subdirectory
            
            # Create and set model metadata with all required fields
            # MTEB's ModelMeta uses Pydantic validation and requires many fields
            self.mteb_model_meta = ModelMeta(
                loader=None,
                name=model_name,  # Must be 'org/model' format
                revision=model_revision,
                release_date=None,
                languages=None,
                n_parameters=None,
                memory_usage_mb=None,
                max_tokens=None,
                embed_dim=None,
                license=None,
                open_weights=None,
                public_training_code=None,
                public_training_data=None,
                framework=[],
                similarity_fn_name=None,
                use_instructions=None,
                training_datasets=None,
            )
            logger.info(f"Set MTEB model metadata: name={model_name}, revision='{model_revision}' (empty)")
        except Exception as e:
            logger.error(f"Failed to set MTEB model metadata: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.mteb_model_meta = None
        
        # Handle device selection
        if device == "cpu":
            self.device = "cpu"
        elif device.startswith("cuda"):
            if torch.cuda.is_available():
                if ":" in device:
                    self.device = device  # e.g., "cuda:1"
                else:
                    # Auto-select GPU with most free memory
                    self.device = select_best_gpu()
            else:
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
        else:
            self.device = "cpu"
        
        logger.info(f"Loading Qwen model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model (offline mode)
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), **load_kwargs)
        logger.info("✓ Tokenizer loaded")
        
        logger.info("Loading model...")
        self.model = AutoModel.from_pretrained(str(model_path), **load_kwargs)
        logger.info("✓ Model loaded")
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model ready. Embedding dimension: {self.model.config.hidden_size}")
        
        # mteb_model_meta is already set before super().__init__()
        # Verify it's still set correctly
        if hasattr(self, 'mteb_model_meta') and self.mteb_model_meta:
            logger.info(f"MTEB model metadata confirmed: name={self.mteb_model_meta.name}, revision={self.mteb_model_meta.revision}")
        else:
            logger.warning("MTEB model metadata not set. Results may use default names.")
    
    def encode(
        self,
        inputs: Union[DataLoader, List[str]],
        *,
        task_metadata=None,
        hf_split: str = None,
        hf_subset: str = None,
        prompt_type=None,
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            inputs: DataLoader (from MTEB) or List[str] (backward compatibility)
            batch_size: Batch size for encoding (used if inputs is List[str])
            **kwargs: Additional arguments (e.g., normalize)
            # Note: task_metadata, hf_split, hf_subset, prompt_type are required by AbsEncoder
            # interface but not used in this implementation
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        # Handle DataLoader (from MTEB)
        if isinstance(inputs, DataLoader):
            all_embeddings = []
            with torch.no_grad():
                for batch in tqdm(inputs, desc="Encoding", leave=False):
                    # Extract texts from batch
                    # Batch can be a list of objects or a dict with 'text' key
                    batch_texts = []
                    if isinstance(batch, dict) and 'text' in batch:
                        # Batch is a dict with 'text' key (list of texts)
                        batch_texts = batch['text']
                    elif isinstance(batch, list):
                        # Batch is a list of TextInput/QueryInput/CorpusInput objects
                        for item in batch:
                            if hasattr(item, 'text'):
                                batch_texts.append(item.text)
                            elif isinstance(item, str):
                                batch_texts.append(item)
                            elif isinstance(item, dict) and 'text' in item:
                                batch_texts.append(item['text'])
                            else:
                                batch_texts.append(str(item))
                    else:
                        logger.warning(f"Unexpected batch type: {type(batch)}")
                        continue
                    
                    if not batch_texts:
                        continue
                    
                    # Tokenize
                    tokenizer_inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    tokenizer_inputs = {k: v.to(self.device) for k, v in tokenizer_inputs.items()}
                    
                    # Get embeddings
                    outputs = self.model(**tokenizer_inputs)
                    embeddings = outputs.last_hidden_state
                    attention_mask = tokenizer_inputs['attention_mask']
                    
                    # Mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                    
                    # Normalize if requested
                    if kwargs.get('normalize', True):
                        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    
                    all_embeddings.append(batch_embeddings.cpu().numpy())
            
            return np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        # Handle List[str] (backward compatibility)
        texts = inputs
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                tokenizer_inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                tokenizer_inputs = {k: v.to(self.device) for k, v in tokenizer_inputs.items()}
                
                # Get embeddings
                outputs = self.model(**tokenizer_inputs)
                embeddings = outputs.last_hidden_state
                attention_mask = tokenizer_inputs['attention_mask']
                
                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # Normalize if requested
                if kwargs.get('normalize', True):
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                all_embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries (same as encode, but can be customized)"""
        return self.encode(queries, **kwargs)
    
    def encode_corpus(self, corpus: List[str], **kwargs) -> np.ndarray:
        """Encode corpus documents (same as encode, but can be customized)"""
        return self.encode(corpus, **kwargs)

