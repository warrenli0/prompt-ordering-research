import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_setup import setup_model
from data_loader import load_dataset_by_name, prefilter_and_sample_examples_multiple, select_in_context_examples_multiple, load_test_set
from prompt_creators import get_prompt_creator
from evaluator import evaluate_model, evaluate_it_model
from utils import reorder_list, generate_random_orderings
from collections import Counter

# Configuration
model_names = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]
token = "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"
dataset_list = ["dbpedia"]  # Example dataset
output_folder = "outputs/combined"
num_samples = 5000
num_test_examples = 500
train_ordering_train_sizes = [100, 500, 1000]  # Size of the training set for train_ordering experiment
num_runs_per_set = 128  # Evaluate each set 10 times
num_sets = 10  # Total of 10 different sets of in-context examples
multiples = [2]  # 2 examples per label
shuffle_seed = 42
label_names = None

def get_text_key(example):
    keys = ['content', 'text', 'sentence', 'question']
    for k in keys:
        if k in example:
            return example[k]
    raise KeyError("No text key found in example")

def get_label_key(example):
    keys = ['label', 'answer']
    for k in keys:
        if k in example:
            return example[k]
    raise KeyError("No label key found in example")

def evaluate(model_name):
    print(f"Running for {model_name}")
    is_instruction_tuned = model_name.endswith('it')
    llm = setup_model(model_name, token)

for model_name in model_names:
    evaluate(model_name)

print("All experiments completed successfully.")