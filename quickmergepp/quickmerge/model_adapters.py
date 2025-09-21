from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForCausalLM,
        LlamaForCausalLM, LlamaTokenizer,
        Qwen2ForCausalLM, Qwen2Tokenizer,
        CLIPTextModel, CLIPTokenizer
    )
    # Diffusion-specific imports
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel
        DIFFUSION_AVAILABLE = True
    except ImportError:
        DIFFUSION_AVAILABLE = False
        AutoencoderKL = None
        UNet2DConditionModel = None
except ImportError:
    # Fallback for environments without transformers
    AutoTokenizer = None
    AutoModel = None
    AutoModelForCausalLM = None
    LlamaForCausalLM = None
    LlamaTokenizer = None
    Qwen2ForCausalLM = None
    Qwen2Tokenizer = None
    CLIPTextModel = None
    CLIPTokenizer = None
    AutoencoderKL = None
    UNet2DConditionModel = None
    DIFFUSION_AVAILABLE = False
import logging
from .pipeline import QuickMergePP
from .multimodal import DiffusionQuickMerge, LLMQuickMerge

logger = logging.getLogger(__name__)


class LLMAdapter(nn.Module):
    """Base adapter for Large Language Models."""
    
    def __init__(self, model_name: str, device: str = "auto", torch_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        raise NotImplementedError
        
    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from the model."""
        raise NotImplementedError
        
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, **kwargs) -> torch.Tensor:
        """Generate text using the model."""
        raise NotImplementedError


class Qwen3Adapter(LLMAdapter):
    """Adapter for Qwen3 models."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", **kwargs):
        super().__init__(model_name, **kwargs)
        
    def load_model(self):
        """Load Qwen3 model and tokenizer."""
        if Qwen2Tokenizer is None or Qwen2ForCausalLM is None:
            raise ImportError("transformers library not available")
        try:
            self.tokenizer = Qwen2Tokenizer.from_pretrained(self.model_name)
            self.model = Qwen2ForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device
            )
            logger.info(f"Loaded Qwen3 model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load Qwen3 model: {e}")
            raise
            
    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from Qwen3 model."""
        if self.model is None:
            self.load_model()
            
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # Return all hidden states from different layers
            hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (num_layers, batch, seq_len, hidden_dim)
            return hidden_states
            
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, **kwargs) -> torch.Tensor:
        """Generate text using Qwen3 model."""
        if self.model is None:
            self.load_model()
            
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                **kwargs
            )
            return outputs


class LLaMAAdapter(LLMAdapter):
    """Adapter for LLaMA models."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", **kwargs):
        super().__init__(model_name, **kwargs)
        
    def load_model(self):
        """Load LLaMA model and tokenizer."""
        if LlamaTokenizer is None or LlamaForCausalLM is None:
            raise ImportError("transformers library not available")
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device
            )
            logger.info(f"Loaded LLaMA model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load LLaMA model: {e}")
            raise
            
    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from LLaMA model."""
        if self.model is None:
            self.load_model()
            
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            return hidden_states
            
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, **kwargs) -> torch.Tensor:
        """Generate text using LLaMA model."""
        if self.model is None:
            self.load_model()
            
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                **kwargs
            )
            return outputs


class DiffusionAdapter(nn.Module):
    """Adapter for Diffusion Models (Stable Diffusion, SDXL)."""
    
    def __init__(
        self, 
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.text_encoder = None
        self.tokenizer = None
        self.unet = None
        self.vae = None
        
    def load_model(self):
        """Load diffusion model components."""
        try:
            # Load text encoder and tokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_name, 
                subfolder="text_encoder",
                torch_dtype=self.torch_dtype,
                device_map=self.device
            )
            
            # Load UNet
            self.unet = UNet2DConditionModel.from_pretrained(
                self.model_name,
                subfolder="unet",
                torch_dtype=self.torch_dtype,
                device_map=self.device
            )
            
            # Load VAE
            self.vae = AutoencoderKL.from_pretrained(
                self.model_name,
                subfolder="vae",
                torch_dtype=self.torch_dtype,
                device_map=self.device
            )
            
            logger.info(f"Loaded diffusion model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load diffusion model: {e}")
            raise
            
    def encode_text(self, text: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text to embeddings."""
        if self.tokenizer is None or self.text_encoder is None:
            self.load_model()
            
        if isinstance(text, str):
            text = [text]
            
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        # Move to device
        tokens = {k: v.to(self.text_encoder.device) for k, v in tokens.items()}
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids)[0]
            
        return text_embeddings, tokens.input_ids
        
    def get_text_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from text encoder."""
        if self.text_encoder is None:
            self.load_model()
            
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            return hidden_states


class UnifiedModelInterface:
    """Unified interface for different model types with QuickMerge++."""
    
    def __init__(
        self,
        model_type: str = "llm",  # "llm", "diffusion", "multimodal"
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        quickmerge_config: Optional[Dict] = None,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Default QuickMerge++ config
        if quickmerge_config is None:
            quickmerge_config = {
                "dim": 4096,
                "k_max": 128,
                "temperature": 0.5
            }
        self.quickmerge_config = quickmerge_config
        
        # Initialize model adapter
        self.adapter = None
        self.quickmerge = None
        self._init_model()
        
    def _init_model(self):
        """Initialize model adapter and QuickMerge++."""
        if self.model_type == "llm":
            if "qwen" in self.model_name.lower() or "qwen2" in self.model_name.lower():
                self.adapter = Qwen3Adapter(self.model_name, self.device, self.torch_dtype)
            elif "llama" in self.model_name.lower():
                self.adapter = LLaMAAdapter(self.model_name, self.device, self.torch_dtype)
            else:
                # Generic LLM adapter
                self.adapter = Qwen3Adapter(self.model_name, self.device, self.torch_dtype)
                
            self.quickmerge = LLMQuickMerge(
                model_dim=self.quickmerge_config["dim"],
                k_max=self.quickmerge_config["k_max"]
            )
            
        elif self.model_type == "diffusion":
            self.adapter = DiffusionAdapter(self.model_name, self.device, self.torch_dtype)
            self.quickmerge = DiffusionQuickMerge(
                unet_dim=768,
                text_dim=768,
                k_max=self.quickmerge_config.get("k_max", 32)
            )
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def load_model(self):
        """Load the model."""
        if self.adapter is not None:
            self.adapter.load_model()
            
    def compress_and_generate(
        self,
        text: str,
        max_new_tokens: int = 50,
        use_compression: bool = True
    ) -> Tuple[str, Dict]:
        """Compress input and generate text."""
        if self.model_type != "llm":
            raise ValueError("compress_and_generate is only supported for LLM models")
            
        if self.adapter is None:
            self.load_model()
            
        # Tokenize input
        input_ids = self.adapter.tokenizer.encode(text, return_tensors="pt")
        if self.device != "cpu":
            input_ids = input_ids.to(self.adapter.model.device)
            
        if use_compression:
            # Get hidden states and compress
            hidden_states = self.adapter.get_hidden_states(input_ids)
            compressed_tokens, info = self.quickmerge.compress(hidden_states)
            
            # Use compressed tokens for generation (simplified approach)
            # In practice, you'd modify the model's forward pass to use compressed tokens
            generated_ids = self.adapter.generate(input_ids, max_new_tokens)
        else:
            # Standard generation without compression
            generated_ids = self.adapter.generate(input_ids, max_new_tokens)
            info = {"compression": False}
            
        # Decode generated text
        generated_text = self.adapter.tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text, info
        
    def compress_text_embeddings(
        self,
        text: str,
        use_compression: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """Compress text embeddings for diffusion models."""
        if self.model_type != "diffusion":
            raise ValueError("compress_text_embeddings is only supported for diffusion models")
            
        if self.adapter is None:
            self.load_model()
            
        # Encode text
        text_embeddings, input_ids = self.adapter.encode_text(text)
        
        if use_compression:
            # Get hidden states and compress
            hidden_states = self.adapter.get_text_hidden_states(input_ids)
            compressed_embeddings, info = self.quickmerge.compress_text_embeddings(
                text_embeddings, hidden_states
            )
            return compressed_embeddings, info
        else:
            return text_embeddings, {"compression": False}


def create_model_interface(
    model_type: str,
    model_name: str,
    **kwargs
) -> UnifiedModelInterface:
    """Factory function to create model interface."""
    return UnifiedModelInterface(
        model_type=model_type,
        model_name=model_name,
        **kwargs
    )
