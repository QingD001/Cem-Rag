"""
Base Embedding Model for MTEB Evaluation

This module provides a base class for embedding models that can be used with MTEB.
Subclasses should implement model-specific logic (e.g., instruction prefixes).
"""

import logging
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils import (
    select_device,
    resolve_huggingface_model_path,
    get_effective_batch_size,
    log_gpu_memory,
)

logger = logging.getLogger(__name__)

# Import AbsEncoder for MTEB compatibility
from mteb.models.abs_encoder import AbsEncoder


class BaseEmbeddingModel(AbsEncoder):
    """Base class for embedding models compatible with MTEB"""
    
    def __init__(
        self,
        model_name_for_hub: str,
        model_path: Path,
        device: str = "cuda",
        project_root: Optional[Path] = None,
    ):
        """
        Initialize base embedding model.
        
        Args:
            model_name_for_hub: HuggingFace model name (e.g., "Qwen/Qwen3-Embedding-0.6B")
            model_path: Path to the model directory
            device: Device to run model on ('cuda', 'cpu', or 'cuda:0', 'cuda:1', etc.)
            project_root: Project root directory (for logging purposes)
        """
        # Call parent class __init__ first
        super().__init__()
        
        self.model_path = model_path
        self.model_name_for_hub = model_name_for_hub
        
        # Set MTEB model metadata
        self._setup_mteb_metadata(model_name_for_hub)
        
        # Handle device selection
        self.device = select_device(device)
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self._load_model()
        
        # Get max sequence length from model config
        self.max_length = getattr(self.model.config, 'max_position_embeddings', 32768)
        logger.info(f"Model ready. Embedding dimension: {self.model.config.hidden_size}, max_length: {self.max_length}")
        
        # Log GPU memory usage after model loading
        if self.device.startswith("cuda"):
            log_gpu_memory(self.device, "GPU memory after model load")
        
        # Verify mteb_model_meta is set correctly
        if hasattr(self, 'mteb_model_meta') and self.mteb_model_meta:
            logger.info(f"MTEB model metadata confirmed: name={self.mteb_model_meta.name}, revision={self.mteb_model_meta.revision}")
        else:
            logger.warning("MTEB model metadata not set. Results may use default names.")
    
    def _setup_mteb_metadata(self, model_name: str):
        """Set up MTEB model metadata."""
        try:
            from mteb.models.model_meta import ModelMeta
            
            self.mteb_model_meta = ModelMeta(
                loader=None,
                name=model_name,
                revision=None,
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
            logger.info(f"Set MTEB model metadata: name={model_name}, revision=None")
        except Exception as e:
            logger.error(f"Failed to set MTEB model metadata: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.mteb_model_meta = None
    
    def _load_model(self):
        """Load tokenizer and model from disk."""
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), **load_kwargs)
        logger.info("✓ Tokenizer loaded")
        
        logger.info("Loading model...")
        self.model = AutoModel.from_pretrained(str(self.model_path), **load_kwargs)
        logger.info("✓ Model loaded")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _extract_texts_from_batch(self, batch) -> list:
        """
        Extract text strings from a batch (from MTEB DataLoader).
        
        Handles different batch formats:
        - dict with 'text' key: batch['text'] is a list of texts
        - list of objects: extract .text attribute or convert to string
        
        Args:
            batch: Batch from MTEB DataLoader (dict or list)
            
        Returns:
            List of text strings
        """
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
        
        return batch_texts
    
    def _preprocess_texts(self, texts: List[str], **kwargs) -> List[str]:
        """
        Preprocess texts before tokenization (e.g., add instruction prefixes).
        
        Subclasses can override this method to add model-specific preprocessing.
        
        Args:
            texts: List of text strings
            **kwargs: Additional arguments (e.g., instruction for E5 models)
            
        Returns:
            Preprocessed list of text strings
        """
        return texts
    
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
            **kwargs: Additional arguments (e.g., normalize, instruction)
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        # Handle DataLoader (from MTEB)
        if isinstance(inputs, DataLoader):
            all_embeddings = []
            with torch.no_grad():
                for batch in tqdm(inputs, desc="Encoding", leave=False):
                    # Extract texts from batch
                    batch_texts = self._extract_texts_from_batch(batch)
                    if not batch_texts:
                        continue
                    
                    # Preprocess texts (e.g., add instruction prefixes)
                    batch_texts = self._preprocess_texts(batch_texts, **kwargs)
                    
                    # For long sequences, use smaller effective batch size to avoid OOM
                    effective_batch_size = get_effective_batch_size(
                        max_length=self.max_length,
                        batch_size=batch_size,
                        batch_texts_length=len(batch_texts)
                    )
                    
                    # Process in smaller chunks if needed
                    for chunk_start in range(0, len(batch_texts), effective_batch_size):
                        chunk_texts = batch_texts[chunk_start:chunk_start + effective_batch_size]
                        
                        # Tokenize and encode
                        chunk_embeddings = self._encode_chunk(chunk_texts, **kwargs)
                        all_embeddings.append(chunk_embeddings.cpu().numpy())
            
            return np.vstack(all_embeddings) if all_embeddings else np.array([])
        
        # Handle List[str] (backward compatibility)
        texts = inputs
        all_embeddings = []
        
        # Preprocess texts (e.g., add instruction prefixes)
        texts = self._preprocess_texts(texts, **kwargs)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize and encode
                batch_embeddings = self._encode_chunk(batch_texts, **kwargs)
                all_embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _encode_chunk(self, texts: List[str], **kwargs) -> torch.Tensor:
        """
        Encode a chunk of texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            **kwargs: Additional arguments (e.g., normalize)
            
        Returns:
            Tensor of embeddings with shape (len(texts), embedding_dim)
        """
        # Tokenize
        tokenizer_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenizer_inputs = {k: v.to(self.device) for k, v in tokenizer_inputs.items()}
        
        # Get embeddings
        outputs = self.model(**tokenizer_inputs)
        embeddings = outputs.last_hidden_state
        attention_mask = tokenizer_inputs['attention_mask']
        
        # Mean pooling and normalize
        embeddings = self._mean_pooling_and_normalize(
            embeddings, attention_mask, normalize=kwargs.get('normalize', True)
        )
        
        return embeddings
    
    def _mean_pooling_and_normalize(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Apply mean pooling to embeddings and optionally normalize.
        
        This is a common operation for transformer-based embedding models.
        
        Args:
            embeddings: Token embeddings from model output, shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            normalize: Whether to L2 normalize the embeddings (default: True)
            
        Returns:
            Pooled (and optionally normalized) embeddings, shape (batch_size, hidden_size)
        """
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_embeddings = sum_embeddings / sum_mask
        
        # Normalize if requested
        if normalize:
            pooled_embeddings = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1)
        
        return pooled_embeddings
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries (can be overridden by subclasses for model-specific behavior)"""
        return self.encode(queries, **kwargs)
    
    def encode_corpus(self, corpus: List[str], **kwargs) -> np.ndarray:
        """Encode corpus documents (can be overridden by subclasses for model-specific behavior)"""
        return self.encode(corpus, **kwargs)
