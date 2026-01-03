"""
E5-Mistral-7B-Instruct Embedding Model Wrapper for MTEB Evaluation

This module provides a wrapper for intfloat/e5-mistral-7b-instruct model.
"""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

from src.models.base_embedding import BaseEmbeddingModel
from src.utils import resolve_huggingface_model_path

logger = logging.getLogger(__name__)


class E5MistralEmbeddingModel(BaseEmbeddingModel):
    """E5-Mistral-7B-Instruct Embedding Model wrapper for MTEB"""
    
    def __init__(self, device: str = "cuda", project_root: Optional[Path] = None):
        """
        Initialize E5-Mistral embedding model.
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or 'cuda:0', 'cuda:1', etc.)
            project_root: Project root directory (default: auto-detect from file location)
        """
        # Determine project root
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        # Model name for HuggingFace
        model_name_for_hub = "intfloat/e5-mistral-7b-instruct"
        base_path = project_root / "models"
        
        # Resolve HuggingFace cache structure to actual model path
        model_path = resolve_huggingface_model_path(base_path, model_name_for_hub)
        
        # Call parent class __init__
        super().__init__(
            model_name_for_hub=model_name_for_hub,
            model_path=model_path,
            device=device,
            project_root=project_root,
        )
    
    def _preprocess_texts(self, texts: List[str], **kwargs) -> List[str]:
        """
        Add instruction prefix to texts for E5 models.
        
        E5 models require instruction prefixes like "query: " or "passage: "
        For retrieval tasks, queries typically use "query: " and documents use "passage: "
        """
        # Determine instruction prefix from kwargs
        # Priority: explicit instruction > prompt_type > default
        instruction = kwargs.get('instruction')
        if instruction is None:
            # Check prompt_type (used by MTEB when calling encode directly)
            prompt_type = kwargs.get('prompt_type', '').lower()
            if 'query' in prompt_type or prompt_type == 'q':
                instruction = 'query: '
            elif 'passage' in prompt_type or prompt_type == 'p' or prompt_type == 'corpus':
                instruction = 'passage: '
            else:
                # Default to query if not specified
                instruction = 'query: '
        
        return [f"{instruction}{text}" for text in texts]
    
    def encode(
        self,
        inputs,
        *,
        task_metadata=None,
        hf_split: str = None,
        hf_subset: str = None,
        prompt_type=None,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Override encode to automatically set instruction prefix based on prompt_type.
        
        MTEB may call encode directly with prompt_type parameter instead of
        using encode_queries/encode_corpus, so we need to handle this case.
        """
        # If prompt_type is provided, set instruction accordingly
        if prompt_type is not None:
            prompt_type_lower = str(prompt_type).lower()
            if 'query' in prompt_type_lower or prompt_type_lower == 'q':
                kwargs['instruction'] = 'query: '
                logger.debug(f"Setting instruction to 'query: ' based on prompt_type={prompt_type}")
            elif 'passage' in prompt_type_lower or prompt_type_lower == 'p' or prompt_type_lower == 'corpus':
                kwargs['instruction'] = 'passage: '
                logger.debug(f"Setting instruction to 'passage: ' based on prompt_type={prompt_type}")
            else:
                logger.warning(f"Unknown prompt_type={prompt_type}, defaulting to 'query: '")
        
        # Call parent encode method
        return super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs
        )
    
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """Encode queries with 'query: ' prefix"""
        return self.encode(queries, instruction="query: ", **kwargs)
    
    def encode_corpus(self, corpus: List[str], **kwargs) -> np.ndarray:
        """Encode corpus documents with 'passage: ' prefix"""
        return self.encode(corpus, instruction="passage: ", **kwargs)
