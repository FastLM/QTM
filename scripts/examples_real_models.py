import torch
import time
from quickmergepp.quickmerge.model_adapters import (
    UnifiedModelInterface, create_model_interface,
    Qwen3Adapter, LLaMAAdapter, DiffusionAdapter
)


def example_qwen3_compression():
    """Example: Qwen3 with QuickMerge++ compression."""
    print("=== Qwen3 with QuickMerge++ ===")
    
    try:
        # Create model interface
        model_interface = create_model_interface(
            model_type="llm",
            model_name="Qwen/Qwen2.5-0.5B",  # Use smaller model for demo
            quickmerge_config={
                "dim": 1024,
                "k_max": 64,
                "temperature": 0.5
            },
            device="cpu"  # Use CPU for demo
        )
        
        # Load model
        print("Loading Qwen3 model...")
        model_interface.load_model()
        
        # Test text
        text = "The future of artificial intelligence is"
        
        # Generate without compression
        print("Generating without compression...")
        start_time = time.time()
        generated_text_no_comp, _ = model_interface.compress_and_generate(
            text, max_new_tokens=20, use_compression=False
        )
        time_no_comp = time.time() - start_time
        
        # Generate with compression
        print("Generating with compression...")
        start_time = time.time()
        generated_text_comp, info = model_interface.compress_and_generate(
            text, max_new_tokens=20, use_compression=True
        )
        time_comp = time.time() - start_time
        
        print(f"Original text: {text}")
        print(f"Generated (no compression): {generated_text_no_comp}")
        print(f"Generated (with compression): {generated_text_comp}")
        print(f"Time without compression: {time_no_comp:.3f}s")
        print(f"Time with compression: {time_comp:.3f}s")
        print(f"Speedup: {time_no_comp / time_comp:.2f}x")
        
    except Exception as e:
        print(f"Error with Qwen3: {e}")
        print("Note: This requires the actual Qwen3 model to be available")


def example_llama_compression():
    """Example: LLaMA with QuickMerge++ compression."""
    print("\n=== LLaMA with QuickMerge++ ===")
    
    try:
        # Create model interface
        model_interface = create_model_interface(
            model_type="llm",
            model_name="meta-llama/Llama-2-7b-hf",
            quickmerge_config={
                "dim": 4096,
                "k_max": 128,
                "temperature": 0.5
            },
            device="cpu"
        )
        
        # Load model
        print("Loading LLaMA model...")
        model_interface.load_model()
        
        # Test text
        text = "Artificial intelligence will transform"
        
        # Generate with compression
        generated_text, info = model_interface.compress_and_generate(
            text, max_new_tokens=15, use_compression=True
        )
        
        print(f"Original text: {text}")
        print(f"Generated: {generated_text}")
        print(f"Compression info: {info}")
        
    except Exception as e:
        print(f"Error with LLaMA: {e}")
        print("Note: This requires the actual LLaMA model to be available")


def example_diffusion_compression():
    """Example: Stable Diffusion with QuickMerge++ compression."""
    print("\n=== Stable Diffusion with QuickMerge++ ===")
    
    try:
        # Create diffusion model interface
        model_interface = create_model_interface(
            model_type="diffusion",
            model_name="runwayml/stable-diffusion-v1-5",
            quickmerge_config={
                "k_max": 32
            },
            device="cpu"
        )
        
        # Load model
        print("Loading Stable Diffusion model...")
        model_interface.load_model()
        
        # Test text prompts
        prompts = [
            "A beautiful landscape with mountains and lakes",
            "A futuristic city with flying cars",
            "A cute cat playing with a ball"
        ]
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            
            # Compress text embeddings
            compressed_embeddings, info = model_interface.compress_text_embeddings(
                prompt, use_compression=True
            )
            
            print(f"Compressed embeddings shape: {compressed_embeddings.shape}")
            print(f"Compression info: {info}")
            
    except Exception as e:
        print(f"Error with Stable Diffusion: {e}")
        print("Note: This requires the actual Stable Diffusion model to be available")


def benchmark_model_compression():
    """Benchmark compression performance across different models."""
    print("\n=== Model Compression Benchmark ===")
    
    models_to_test = [
        {
            "type": "llm",
            "name": "Qwen/Qwen2.5-0.5B",
            "config": {"dim": 1024, "k_max": 64}
        },
        {
            "type": "diffusion", 
            "name": "runwayml/stable-diffusion-v1-5",
            "config": {"k_max": 32}
        }
    ]
    
    test_text = "The future of artificial intelligence is bright and promising."
    
    for model_config in models_to_test:
        print(f"\nTesting {model_config['name']}...")
        
        try:
            model_interface = create_model_interface(
                model_type=model_config["type"],
                model_name=model_config["name"],
                quickmerge_config=model_config["config"],
                device="cpu"
            )
            
            # Load model
            start_time = time.time()
            model_interface.load_model()
            load_time = time.time() - start_time
            
            if model_config["type"] == "llm":
                # Test LLM compression
                start_time = time.time()
                generated_text, info = model_interface.compress_and_generate(
                    test_text, max_new_tokens=10, use_compression=True
                )
                inference_time = time.time() - start_time
                
                print(f"Load time: {load_time:.3f}s")
                print(f"Inference time: {inference_time:.3f}s")
                print(f"Generated: {generated_text}")
                
            elif model_config["type"] == "diffusion":
                # Test diffusion compression
                start_time = time.time()
                compressed_embeddings, info = model_interface.compress_text_embeddings(
                    test_text, use_compression=True
                )
                compression_time = time.time() - start_time
                
                print(f"Load time: {load_time:.3f}s")
                print(f"Compression time: {compression_time:.3f}s")
                print(f"Compressed shape: {compressed_embeddings.shape}")
                
        except Exception as e:
            print(f"Error testing {model_config['name']}: {e}")


def example_custom_adapters():
    """Example: Using custom adapters directly."""
    print("\n=== Custom Adapters Example ===")
    
    # Test with mock data (no actual model loading)
    print("Testing adapter interfaces with mock data...")
    
    # Mock hidden states
    batch_size, seq_len, hidden_dim = 2, 128, 1024
    mock_hidden_states = torch.randn(4, batch_size, seq_len, hidden_dim)
    
    # Test QuickMerge++ compression
    from quickmergepp.quickmerge.multimodal import LLMQuickMerge
    
    llm_quickmerge = LLMQuickMerge(
        model_dim=hidden_dim,
        k_max=64
    )
    
    compressed_tokens, info = llm_quickmerge.adaptive_compression(mock_hidden_states[-1])
    
    print(f"Original shape: {mock_hidden_states.shape}")
    print(f"Compressed shape: {compressed_tokens.shape}")
    print(f"Compression ratio: {mock_hidden_states.size(1) / compressed_tokens.size(1):.2f}x")


if __name__ == "__main__":
    # Run examples
    example_qwen3_compression()
    example_llama_compression()
    example_diffusion_compression()
    benchmark_model_compression()
    example_custom_adapters()
