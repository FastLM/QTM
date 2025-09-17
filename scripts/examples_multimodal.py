import torch
import torch.nn as nn
from quickmergepp.quickmerge.multimodal import (
    VisionEncoder, VideoEncoder, MultimodalQuickMerge,
    DiffusionQuickMerge, LLMQuickMerge,
    create_multimodal_pipeline, create_diffusion_pipeline, create_llm_pipeline
)


def example_vision_language():
    """Example: Vision-Language with QuickMerge++."""
    print("=== Vision-Language Example ===")
    
    # Create components
    vision_encoder = VisionEncoder(img_size=224, patch_size=16, dim=768)
    multimodal_pipeline = create_multimodal_pipeline(
        vision_dim=768, text_dim=512, compressed_dim=256, k_max=64
    )
    
    # Mock data
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    text_tokens = torch.randn(batch_size, 50, 512)
    
    # Process
    vision_tokens = vision_encoder(images)  # (batch, 197, 768)
    print(f"Original vision tokens: {vision_tokens.shape}")
    print(f"Original text tokens: {text_tokens.shape}")
    
    # Compress and fuse
    fused_output, info = multimodal_pipeline(
        vision_tokens=vision_tokens,
        text_tokens=text_tokens
    )
    
    print(f"Fused output: {fused_output.shape}")
    print(f"Compression info: {info}")


def example_video_text():
    """Example: Video-Text with QuickMerge++."""
    print("\n=== Video-Text Example ===")
    
    # Create components
    video_encoder = VideoEncoder(img_size=224, patch_size=16, num_frames=16, dim=768)
    multimodal_pipeline = create_multimodal_pipeline(
        vision_dim=768, text_dim=512, compressed_dim=256, k_max=32
    )
    
    # Mock data
    batch_size = 1
    videos = torch.randn(batch_size, 3, 16, 224, 224)  # (B, C, T, H, W)
    text_tokens = torch.randn(batch_size, 100, 512)
    
    # Process
    video_tokens = video_encoder(videos)  # (batch, 4097, 768) - 16*16*16 + 1 cls
    print(f"Original video tokens: {video_tokens.shape}")
    print(f"Original text tokens: {text_tokens.shape}")
    
    # Compress and fuse
    fused_output, info = multimodal_pipeline(
        vision_tokens=video_tokens,
        text_tokens=text_tokens
    )
    
    print(f"Fused output: {fused_output.shape}")
    print(f"Compression ratio: {video_tokens.shape[1] / fused_output.shape[1]:.2f}x")


def example_diffusion_model():
    """Example: Diffusion Model with QuickMerge++."""
    print("\n=== Diffusion Model Example ===")
    
    # Create diffusion pipeline
    diffusion_pipeline = create_diffusion_pipeline(
        unet_dim=768, text_dim=512, k_max=32
    )
    
    # Mock data
    batch_size = 2
    text_embeddings = torch.randn(batch_size, 77, 512)  # CLIP-like
    unet_features = torch.randn(4, batch_size, 77, 768)  # Multi-layer UNet features
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    print(f"Original text embeddings: {text_embeddings.shape}")
    print(f"Original UNet features: {unet_features.shape}")
    
    # Compress text embeddings
    compressed_text, text_info = diffusion_pipeline.compress_text_embeddings(text_embeddings)
    print(f"Compressed text: {compressed_text.shape}")
    
    # Compress UNet features
    compressed_unet, time_emb = diffusion_pipeline.compress_unet_features(unet_features, timesteps)
    print(f"Compressed UNet: {compressed_unet.shape}")
    print(f"Time embedding: {time_emb.shape}")


def example_llm_optimization():
    """Example: LLM with QuickMerge++ optimizations."""
    print("\n=== LLM Optimization Example ===")
    
    # Create LLM pipeline
    llm_pipeline = create_llm_pipeline(
        model_dim=4096, vocab_size=50000, k_max=128, max_seq_len=2048
    )
    
    # Mock data
    batch_size = 1
    seq_len = 1024
    hidden_states = torch.randn(4, batch_size, seq_len, 4096)  # Multi-layer
    key_cache = torch.randn(batch_size, seq_len, 32, 128)  # (B, S, H, D)
    value_cache = torch.randn(batch_size, seq_len, 32, 128)
    
    print(f"Original hidden states: {hidden_states.shape}")
    print(f"Original KV cache: {key_cache.shape}, {value_cache.shape}")
    
    # Adaptive compression
    compressed_tokens, info = llm_pipeline.adaptive_compression(hidden_states[-1])
    print(f"Compressed tokens: {compressed_tokens.shape}")
    
    # KV cache compression
    compressed_k, compressed_v = llm_pipeline.compress_kv_cache(key_cache, value_cache)
    print(f"Compressed KV: {compressed_k.shape}, {compressed_v.shape}")
    
    # Memory savings
    original_memory = key_cache.numel() + value_cache.numel()
    compressed_memory = compressed_k.numel() + compressed_v.numel()
    print(f"Memory reduction: {original_memory / compressed_memory:.2f}x")


def example_speculative_multimodal():
    """Example: Speculative Decoding with Multimodal inputs."""
    print("\n=== Speculative Multimodal Example ===")
    
    from quickmergepp.quickmerge.speculative import create_speculative_decoder
    
    # Mock multimodal target model
    class MultimodalTargetModel(nn.Module):
        def __init__(self, vocab_size=1000, dim=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, dim)
            self.transformer = nn.TransformerDecoderLayer(
                d_model=dim, nhead=8, dim_feedforward=dim*4, batch_first=True
            )
            self.head = nn.Linear(dim, vocab_size)
            
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.transformer(x, x)
            return self.head(x)
        
        def get_last_hidden_states(self):
            return torch.randn(4, 2, 128, 512)
    
    # Create speculative decoder
    target_model = MultimodalTargetModel()
    spec_decoder = create_speculative_decoder(
        target_model=target_model,
        vocab_size=1000,
        dim=512,
        quickmerge_dim=256,
        k_max=32,
        max_draft_len=4
    )
    
    # Generate with multimodal context
    input_ids = torch.randint(0, 1000, (2, 64))
    hidden_states = target_model.get_last_hidden_states()
    
    generated_ids, stats = spec_decoder.generate(
        input_ids=input_ids,
        hidden_states=hidden_states,
        max_new_tokens=20,
        verbose=False
    )
    
    print(f"Generated sequence length: {generated_ids.shape[1]}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"Average speedup: {stats['avg_speedup']:.2f}x")


if __name__ == "__main__":
    example_vision_language()
    example_video_text()
    example_diffusion_model()
    example_llm_optimization()
    example_speculative_multimodal()
