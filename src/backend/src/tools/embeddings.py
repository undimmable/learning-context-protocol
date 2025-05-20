"""Text to embeddings conversion for the AGI system.

This module provides functions for converting text to embeddings using
various models. It's used by the Vector Database Provider and other
components that need to work with embeddings.

Usage:
    from src.tools.embeddings import get_embedding

    # Get an embedding for a text
    embedding = get_embedding("Hello, world!")
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# Embeddings models
# ---------------------------------------------------------------------------

# Default model to use for embeddings
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def _get_sentence_transformer():
    """Get a sentence transformer model for generating embeddings.

    This is cached so subsequent calls are quick. We use the all-MiniLM-L6-v2
    model by default, which is a good balance between quality and performance.

    Returns:
        A sentence transformer model.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    except ImportError:
        raise ImportError(
            "The sentence-transformers library is required for embeddings, but it "
            "is not installed in this environment. Please install it with: "
            "pip install sentence-transformers"
        )

def get_embedding(
    text: Union[str, List[str]],
    model: Optional[str] = None,
) -> Union[List[float], List[List[float]]]:
    """Get an embedding for a text or list of texts.

    Args:
        text: The text or list of texts to get embeddings for.
        model: The model to use for embeddings. If None, uses the default model.

    Returns:
        A list of floats (the embedding) or a list of lists of floats (for multiple texts).
    """
    # If a specific model is requested, we'll use it instead of the cached one
    if model and model != DEFAULT_EMBEDDING_MODEL:
        try:
            from sentence_transformers import SentenceTransformer
            model_instance = SentenceTransformer(model)
            embeddings = model_instance.encode(text, convert_to_numpy=False)
            return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        except ImportError:
            raise ImportError(
                "The sentence-transformers library is required for embeddings, but it "
                "is not installed in this environment. Please install it with: "
                "pip install sentence-transformers"
            )
    
    # Use the cached model for the default case
    model_instance = _get_sentence_transformer()
    embeddings = model_instance.encode(text, convert_to_numpy=False)
    return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings

# ---------------------------------------------------------------------------
# Embedding similarity functions
# ---------------------------------------------------------------------------

def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate the cosine similarity between two embeddings.

    Args:
        embedding1: The first embedding.
        embedding2: The second embedding.

    Returns:
        The cosine similarity between the embeddings (between -1 and 1).
    """
    import numpy as np
    
    # Convert to numpy arrays if they aren't already
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)