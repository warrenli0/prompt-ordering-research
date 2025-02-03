import numpy as np
from huggingface_hub import login
from vllm import LLM

def setup_model(model_name, token):
    login(token=token)
    llm = LLM(model=model_name, 
              gpu_memory_utilization=0.9, 
              tensor_parallel_size=1,
              enforce_eager=True)
    return llm
