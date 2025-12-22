"""
Layout-aware embedding using ColPali (SigLIP + Gemma-2B).

ColPali combines a vision encoder (SigLIP) with a language model projection (Gemma)
for better instruction-following in visual document retrieval tasks.

Compatible with colpali-engine >= 0.3.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import warnings


class ColPaliEmbedder(nn.Module):
    """
    Layout-Aware Embedding using ColPali (SigLIP + Gemma-2B).
    
    Architecture:
        - Vision Tower: SigLIP-400M (ViT for image patches)
        - Text Tower: Gemma-2B projection (NOT SigLIP text encoder)
        - Output: Multi-vector embeddings (averaged to 128-dim)
    
    Key Difference from CLIP:
        ColPali uses a language model (Gemma) for text encoding instead of
        a simple text encoder, allowing better instruction-following and
        understanding of complex queries.
    
    Reference: 
        ColPali: Efficient Document Retrieval with Vision Language Models
        https://arxiv.org/abs/2407.01449
        
    Example:
        >>> embedder = ColPaliEmbedder()
        >>> images = torch.rand(10, 3, 224, 224)
        >>> patch_embs = embedder.encode_patches(images)
        >>> query_emb = embedder.encode_query("Find technical diagrams")
        >>> similarities = patch_embs @ query_emb.T
    """
    
    def __init__(
        self, 
        model_name: str = "vidore/colpali-v1.2", 
        device: Optional[str] = None
    ):
        """
        Initialize ColPali embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Target device ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self._load_model(model_name)
    
    def _load_model(self, model_name: str) -> None:
        """
        Load ColPali model and processor (compatible with v0.3+).
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"🧠 Loading ColPali model: {model_name}")
        print(f"   Device: {self.device}")
        
        try:
            # Import ColPali components (v0.3+ API)
            from colpali_engine.models import ColPali, ColPaliProcessor
            
            # Load model with appropriate dtype
            self.model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map=self.device
            ).eval()
            
            # Load processor
            self.processor = ColPaliProcessor.from_pretrained(model_name)
            
            print(f"✓ ColPali loaded successfully")
            print(f"   Model dtype: {next(self.model.parameters()).dtype}")
            
        except ImportError as e:
            warnings.warn(
                f"ColPali import failed: {e}\n"
                f"Make sure colpali-engine is installed: pip install colpali-engine\n"
                f"Falling back to dummy embeddings for testing."
            )
            self.model = None
            self.processor = None
            
        except Exception as e:
            warnings.warn(
                f"Failed to load ColPali model: {e}\n"
                f"Falling back to dummy embeddings for testing."
            )
            self.model = None
            self.processor = None
    
    def encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode image patches using ColPali vision tower.
        
        Args:
            images: Tensor of shape (N, C, H, W) normalized to [0, 1]
        
        Returns:
            Embeddings of shape (N, 128) - L2-normalized
        
        Process:
            1. Resize to 224x224 if needed (ColPali expects this size)
            2. Convert tensors to PIL Images
            3. Process through ColPali processor
            4. Get multi-vector embeddings from model
            5. Average pool to single vector
            6. Project to 128-dim and L2-normalize
        """
        if self.model is None:
            # Fallback: return random normalized embeddings
            N = images.shape[0]
            embeddings = torch.randn(N, 128, device=images.device)
            return F.normalize(embeddings, p=2, dim=1)
        
        # Resize to 224x224 if needed (ColPali's expected input size)
        if images.shape[2] != 224 or images.shape[3] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            # Convert tensors to PIL Images
            from PIL import Image
            import numpy as np
            
            pil_images = []
            for i in range(images.shape[0]):
                # Convert (C, H, W) tensor to PIL Image
                img_np = images[i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            
            # Process images with ColPali processor
            batch_images = self.processor.process_images(pil_images)
            
            # Move to device
            batch_images = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_images.items()
            }
            
            # Get embeddings from model (returns multi-vector)
            embeddings = self.model(**batch_images)
            
            # ColPali returns multi-vector representations
            # Shape: (batch_size, seq_len, hidden_dim)
            if len(embeddings.shape) == 3:
                # Average pool over sequence length
                embeddings = embeddings.mean(dim=1)  # (batch_size, hidden_dim)
            
            # Project to 128-dim if needed
            if embeddings.shape[1] != 128:
                if not hasattr(self, 'projection'):
                    self.projection = nn.Linear(
                        embeddings.shape[1], 128, 
                        device=self.device,
                        dtype=embeddings.dtype
                    )
                embeddings = self.projection(embeddings)
            
            # L2 normalize for cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Convert to float32
            embeddings = embeddings.to(torch.float32)
        
        return embeddings
    
    def encode_query(self, text: str) -> torch.Tensor:
        """
        Encode text query using Gemma-2B projection layer.
        
        Args:
            text: Query string (e.g., "Find all figures about neural networks")
        
        Returns:
            Query embedding of shape (1, 128) - L2-normalized
        
        CRITICAL IMPLEMENTATION NOTE:
            This uses the Gemma language model projection, NOT the SigLIP 
            text encoder. This is the key innovation in ColPali - the query 
            is processed through a full language model for better semantic 
            understanding and instruction-following capabilities.
            
        Example Queries:
            - "Find all mathematical equations in Section 3"
            - "Locate diagrams showing system architecture"
            - "Retrieve tables with experimental results"
        """
        if self.model is None:
            # Fallback: return random normalized embedding
            embedding = torch.randn(1, 128, device=self.device)
            return F.normalize(embedding, p=2, dim=1)
        
        with torch.no_grad():
            # Process query with ColPali processor
            batch_queries = self.processor.process_queries([text])
            
            # Move to device
            batch_queries = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch_queries.items()
            }
            
            # Get query embedding (returns multi-vector)
            query_embedding = self.model(**batch_queries)
            
            # Handle multi-vector output
            if len(query_embedding.shape) == 3:
                # Average pool over sequence length
                query_embedding = query_embedding.mean(dim=1)  # (1, hidden_dim)
            
            # Project to 128-dim if needed
            if query_embedding.shape[1] != 128:
                if not hasattr(self, 'projection'):
                    self.projection = nn.Linear(
                        query_embedding.shape[1], 128,
                        device=self.device,
                        dtype=query_embedding.dtype
                    )
                query_embedding = self.projection(query_embedding)
            
            # L2 normalize
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
            # Convert to float32
            query_embedding = query_embedding.to(torch.float32)
        
        return query_embedding
    
    def forward(
        self, 
        images: Optional[torch.Tensor] = None,
        query: Optional[str] = None
    ) -> torch.Tensor:
        """
        Forward pass supporting both image and text encoding.
        
        Args:
            images: Optional tensor of shape (N, C, H, W)
            query: Optional query string
        
        Returns:
            Embeddings tensor
        
        Raises:
            ValueError: If neither images nor query is provided
        """
        if images is not None:
            return self.encode_patches(images)
        elif query is not None:
            return self.encode_query(query)
        else:
            raise ValueError("Must provide either images or query")