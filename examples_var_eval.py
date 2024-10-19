import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_setup import setup_model
from data_loader import load_dataset_by_name, prefilter_and_sample_examples_multiple, select_in_context_examples_multiple
from prompt_creators import get_prompt_creator
from evaluator import evaluate_model
from utils import reorder_list

# Configuration
model_name = "google/gemma-2b"
token = "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"
dataset_list = ["nyt-topics", "nyt-locations", "sst2", "ag_news", "dbpedia"]
num_samples = 500
num_test_examples = 200
num_runs = 100  # We need 100 random sets
multiples = [2]  # 2 examples per label
shuffle_seed = 42
label_names = None

# Setup
llm = setup_model(model_name, token)

# Get the correct key for text/content
def get_text_key(example):
    # Return the appropriate key based on what exists in the example
    if 'content' in example:
        return example['content']
    elif 'text' in example:
        return example['text']
    elif 'sentence' in example:
        return example['sentence']
    else:
        raise KeyError("Neither 'content' nor 'text' key is found in the example")

for dataset_name in dataset_list:
    print(f"Running for {dataset_name}")

    # Load dataset
    train_set, test_set, num_classes, label_names = load_dataset_by_name(dataset_name, num_samples)
    test_data = test_set.shuffle(seed=42).select(range(num_test_examples))

    # Prefilter, shuffle, and sample examples for each label
    total_examples_per_class = num_classes * 5  # Store 5x more examples than needed
    sampled_data = prefilter_and_sample_examples_multiple(train_set, num_classes, total_examples_per_class, seed=shuffle_seed)

    # Get the correct prompt creation function
    create_prompt_fn = get_prompt_creator(dataset_name, label_names)

    # Create output directories
    output_dirs = ['outputs/results/random_example', 'outputs/boxplots', 'outputs/frequency_graphs']
    for dir_name in output_dirs:
        os.makedirs(dir_name, exist_ok=True)

    # Initialize containers for storing results
    all_accuracies = []
    ordering_accuracies = {}

    # Iterate over each multiple
    for multiple in multiples:
        num_incontext_examples = num_classes * multiple

        # Run experiment for each set
        for run in tqdm(range(num_runs)):
            # Fix random seed for deterministic results
            random.seed(run + shuffle_seed)

             # Dynamically sample in-context examples during each run
            in_context_data = select_in_context_examples_multiple(sampled_data, num_classes, num_incontext_examples, seed=run + shuffle_seed)

            # Sort by label and within each label alphabetically by text
            in_context_data = sorted(in_context_data, key=lambda x: (x['label'], get_text_key(x)))

            # Generate fixed permutation (deterministic)
            ordering = list(range(len(in_context_data)))

            # Evaluate the model
            accuracy, _ = evaluate_model(llm, dataset_name, test_data, reorder_list(in_context_data, ordering), create_prompt_fn, label_names)
            all_accuracies.append(accuracy)

            # Store ordering-wise accuracy
            ordering_accuracies.setdefault(run, []).append(accuracy)

    # Save accuracies to CSV
    accuracy_df = pd.DataFrame(ordering_accuracies)
    accuracy_df.to_csv(f'outputs/results/random_example/{dataset_name}_accuracy_results.csv', index=False)

    # Create box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_accuracies)
    plt.title(f'Accuracy Distribution for {dataset_name}')
    plt.ylabel('Accuracy')
    plt.savefig(f'outputs/boxplots/{dataset_name}_random_example_accuracy_boxplot.png')
    plt.close()

    # Create frequency graph (histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(all_accuracies, bins=20, edgecolor='black')
    plt.title(f'Accuracy Frequency for {dataset_name}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.savefig(f'outputs/frequency_graphs/{dataset_name}_random_example_accuracy_histogram.png')
    plt.close()

print("Experiments completed for all datasets.")
