from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pipeline import QuickMergePP
from .speculative import SpeculativeDecoder


class VisionEncoder(nn.Module):
    """Vision encoder with patch-based tokenization."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, dim)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x


class VideoEncoder(nn.Module):
    """Video encoder with temporal-spatial tokenization."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, num_frames: int = 16, dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.spatial_patches = (img_size // patch_size) ** 2
        self.total_patches = self.spatial_patches * num_frames
        
        self.patch_embed = nn.Conv3d(3, dim, kernel_size=(3, patch_size, patch_size), 
                                    stride=(1, patch_size, patch_size), padding=(1, 0, 0))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # x: (B, C, T, H, W)
        x = self.patch_embed(x)  # (B, dim, T, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, total_patches, dim)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x


class MultimodalQuickMerge(nn.Module):
    """Multimodal QuickMerge++ for vision-language tasks."""
    
    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 512,
        compressed_dim: int = 256,
        k_max: int = 64,
        num_modalities: int = 2
    ):
        super().__init__()
        self.vision_compressor = QuickMergePP(dim=vision_dim, k_max=k_max)
        self.text_compressor = QuickMergePP(dim=text_dim, k_max=k_max)
        
        # Cross-modal fusion
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=compressed_dim, num_heads=8, batch_first=True
        )
        self.projection = nn.Linear(vision_dim + text_dim, compressed_dim)
        
    def forward(
        self, 
        vision_tokens: torch.Tensor, 
        text_tokens: torch.Tensor,
        vision_hidden_states: Optional[torch.Tensor] = None,
        text_hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            vision_tokens: (batch, seq_len, vision_dim)
            text_tokens: (batch, seq_len, text_dim)
            vision_hidden_states: (num_layers, batch, seq_len, vision_dim)
            text_hidden_states: (num_layers, batch, seq_len, text_dim)
        """
        # Compress each modality
        if vision_hidden_states is not None:
            compressed_vision, vision_info = self.vision_compressor.compress(vision_hidden_states)
        else:
            compressed_vision = vision_tokens
            vision_info = {}
            
        if text_hidden_states is not None:
            compressed_text, text_info = self.text_compressor.compress(text_hidden_states)
        else:
            compressed_text = text_tokens
            text_info = {}
        
        # Cross-modal fusion - handle different sequence lengths
        min_seq_len = min(compressed_vision.size(1), compressed_text.size(1))
        compressed_vision = compressed_vision[:, :min_seq_len, :]
        compressed_text = compressed_text[:, :min_seq_len, :]
        
        fused_tokens = torch.cat([compressed_vision, compressed_text], dim=-1)
        fused_tokens = self.projection(fused_tokens)
        
        # Self-attention for fusion
        fused_output, _ = self.fusion_layer(fused_tokens, fused_tokens, fused_tokens)
        
        info = {
            'vision_info': vision_info,
            'text_info': text_info,
            'compressed_vision_shape': compressed_vision.shape,
            'compressed_text_shape': compressed_text.shape
        }
        
        return fused_output, info


class DiffusionQuickMerge(nn.Module):
    """QuickMerge++ integration for Diffusion Models."""
    
    def __init__(
        self,
        unet_dim: int = 768,
        text_dim: int = 512,
        k_max: int = 32,
        num_timesteps: int = 1000
    ):
        super().__init__()
        self.text_compressor = QuickMergePP(dim=text_dim, k_max=k_max)
        self.unet_compressor = QuickMergePP(dim=unet_dim, k_max=k_max)
        
        # Time embedding compression
        self.time_embed = nn.Sequential(
            nn.Linear(1, unet_dim),
            nn.SiLU(),
            nn.Linear(unet_dim, unet_dim)
        )
        
    def compress_text_embeddings(
        self, 
        text_embeddings: torch.Tensor,
        text_hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Compress text embeddings for diffusion conditioning."""
        if text_hidden_states is not None:
            compressed_text, info = self.text_compressor.compress(text_hidden_states)
        else:
            # Use text embeddings directly
            compressed_text = text_embeddings
            info = {}
        
        return compressed_text, info
    
    def compress_unet_features(
        self,
        unet_features: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress UNet features with time conditioning."""
        # Add time embedding
        time_emb = self.time_embed(timesteps.float().unsqueeze(-1))
        
        # Compress UNet features
        compressed_features, _ = self.unet_compressor.compress(unet_features)
        
        return compressed_features, time_emb


class LLMQuickMerge(nn.Module):
    """QuickMerge++ optimizations for Large Language Models."""
    
    def __init__(
        self,
        model_dim: int = 4096,
        vocab_size: int = 50000,
        k_max: int = 128,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.compressor = QuickMergePP(dim=model_dim, k_max=k_max)
        self.max_seq_len = max_seq_len
        
        # KV cache compression
        self.kv_compressor = QuickMergePP(dim=model_dim, k_max=k_max//2)
        
    def compress_kv_cache(
        self, 
        key_cache: torch.Tensor, 
        value_cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress KV cache to reduce memory usage."""
        # Reshape for compression: (batch, seq_len, num_heads, head_dim)
        B, S, H, D = key_cache.shape
        
        # Flatten to (batch, seq_len, num_heads * head_dim)
        k_flat = key_cache.view(B, S, H * D)
        v_flat = value_cache.view(B, S, H * D)
        
        # Compress
        compressed_k, _ = self.kv_compressor.compress(k_flat.unsqueeze(0))
        compressed_v, _ = self.kv_compressor.compress(v_flat.unsqueeze(0))
        
        # Reshape back
        compressed_k = compressed_k.view(B, -1, H, D)
        compressed_v = compressed_v.view(B, -1, H, D)
        
        return compressed_k, compressed_v
    
    def adaptive_compression(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Adaptive compression based on sequence length and attention patterns."""
        seq_len = hidden_states.size(1)
        
        # Dynamic k_max based on sequence length
        if seq_len > self.max_seq_len // 2:
            k_max = self.compressor.k_max
        else:
            k_max = min(seq_len // 2, self.compressor.k_max)
        
        # Create temporary compressor with adjusted k_max
        temp_compressor = QuickMergePP(dim=hidden_states.size(-1), k_max=k_max)
        compressed_tokens, info = temp_compressor.compress(hidden_states.unsqueeze(0))
        
        return compressed_tokens.squeeze(0), info


def create_multimodal_pipeline(
    vision_dim: int = 768,
    text_dim: int = 512,
    compressed_dim: int = 256,
    k_max: int = 64
) -> MultimodalQuickMerge:
    """Factory function for multimodal QuickMerge++."""
    return MultimodalQuickMerge(
        vision_dim=vision_dim,
        text_dim=text_dim,
        compressed_dim=compressed_dim,
        k_max=k_max
    )


def create_diffusion_pipeline(
    unet_dim: int = 768,
    text_dim: int = 512,
    k_max: int = 32
) -> DiffusionQuickMerge:
    """Factory function for diffusion QuickMerge++."""
    return DiffusionQuickMerge(
        unet_dim=unet_dim,
        text_dim=text_dim,
        k_max=k_max
    )


def create_llm_pipeline(
    model_dim: int = 4096,
    vocab_size: int = 50000,
    k_max: int = 128,
    max_seq_len: int = 2048
) -> LLMQuickMerge:
    """Factory function for LLM QuickMerge++."""
    return LLMQuickMerge(
        model_dim=model_dim,
        vocab_size=vocab_size,
        k_max=k_max,
        max_seq_len=max_seq_len
    )
