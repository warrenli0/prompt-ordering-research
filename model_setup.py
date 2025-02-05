import numpy as np
from huggingface_hub import login
from vllm import LLM
from transformers import AutoConfig

def get_model_max_length(model_name, default=128000):
    # Load the configuration of the model.
    config = AutoConfig.from_pretrained(model_name)
    
    # Try common attributes. Adjust the order if needed.
    if hasattr(config, "max_position_embeddings") and config.max_position_embeddings:
        return config.max_position_embeddings
    elif hasattr(config, "model_max_length") and config.model_max_length:
        return config.model_max_length
    else:
        # Fallback: use a default value.
        return default

def setup_model(model_name, token):
    login(token=token)

    # Automatically get the model's max length from its config.
    derived_max_model_len = get_model_max_length(model_name)
    # You might want to use a user desired value if it’s lower than the derived one.
    # For example, if you wish to use up to 128000 tokens, but the model only supports less:
    desired_max_model_len = min(128000, derived_max_model_len)
    print(f"Using max_model_len = {desired_max_model_len} for {model_name}")
    
    llm = LLM(model=model_name, 
              gpu_memory_utilization=0.95, 
              max_model_len=desired_max_model_len,
              tensor_parallel_size=1,
              dtype="bfloat16",
              enforce_eager=True)
    return llm
