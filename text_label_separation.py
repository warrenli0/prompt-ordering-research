import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from model_setup import setup_model
from data_loader import load_dataset_by_name, prefilter_and_sample_examples_multiple, select_in_context_examples_multiple
from prompt_creators import get_prompt_creator
from evaluator import evaluate_model, evaluate_it_model
from utils import reorder_list, generate_random_orderings

# Configuration
model_name = "gpt2-xl"
is_instruction_tuned = model_name.endswith('it')
print("Instruction tuned?", is_instruction_tuned)
token = "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"
dataset_list = ["nyt-topics", "nyt-locations", "sst2", "ag_news", "dbpedia", "mmlu"]
output_folder = "outputs/text_label_separation"
num_samples = 500
num_test_examples = 200
num_runs_per_set = 10  # We now evaluate each set 10 times
num_sets = 10  # Total of 10 different sets of in-context examples
multiples = [2]  # 2 examples per label
shuffle_seed = 42
label_names = None

# Setup
llm = setup_model(model_name, token)

# Get the correct key for text/content
def get_text_key(example):
    if 'content' in example:
        return 'content'
    elif 'text' in example:
        return 'text'
    elif 'sentence' in example:
        return 'sentence'
    elif 'question' in example:
        return 'question'
    else:
        raise KeyError("Neither 'content' nor 'text' key is found in the example")

# Get the correct key for label
def get_label_key(example):
    if 'label' in example:
        return 'label'
    elif 'answer' in example:
        return 'answer'
    else:
        raise KeyError("Neither key is found in the example")

for dataset_name in dataset_list:
    print(f"Running for {dataset_name}")

    label_key = 'answer' if dataset_name == 'mmlu' else 'label'

    # Load dataset
    train_set, test_set, num_classes, label_names = load_dataset_by_name(dataset_name, num_samples)
    test_data = test_set.shuffle(seed=42).select(range(num_test_examples))

    # Prefilter, shuffle, and sample examples for each label
    total_examples_per_class = num_classes * 5  # Store 5x more examples than needed
    sampled_data = prefilter_and_sample_examples_multiple(train_set, num_classes, total_examples_per_class, seed=shuffle_seed)

    # Get the correct prompt creation function
    create_prompt_fn = get_prompt_creator(dataset_name, is_instruction_tuned=is_instruction_tuned, label_names=label_names)

    # Create output directories
    output_dirs = [
        f'{output_folder}/{model_name}/results', 
        f'{output_folder}/{model_name}/boxplots', 
        f'{output_folder}/{model_name}/frequency_graphs',
        f'{output_folder}/{model_name}/generated',
    ]
    for dir_name in output_dirs:
        os.makedirs(dir_name, exist_ok=True)

    # Initialize containers for storing results
    all_accuracies = []
    ordering_accuracies = {}

    # Iterate over each multiple
    for multiple in multiples:
        num_incontext_examples = num_classes * multiple

        # Generate the 10 random sets of in-context examples only once
        in_context_sets = []
        for set_run in range(num_sets):
            in_context_data = select_in_context_examples_multiple(sampled_data, num_classes, num_incontext_examples, seed=set_run + shuffle_seed)
            in_context_data = sorted(in_context_data, key=lambda x: (x[label_key], x[get_text_key(x)]))
            in_context_sets.append(in_context_data)  # Store each set

        # Generate 10 different sets of in-context examples
        for set_run in tqdm(range(num_sets)):
            # Dynamically sample in-context examples for this set
            in_context_data = in_context_sets[set_run]

            # Sort by label and within each label alphabetically by text
            in_context_data = sorted(in_context_data, key=lambda x: (x[label_key], x[get_text_key(x)]))

            # Generate 10 unique orderings per in-context example set
            orderings = generate_random_orderings([0] * num_incontext_examples, num_orderings=num_runs_per_set)

            # Evaluate this set using each of the 10 permutations
            for perm_run in range(num_runs_per_set):
                # Extract texts and apply the permutation
                texts = [x[get_text_key(x)] for x in in_context_data]
                permuted_texts = reorder_list(texts, orderings[perm_run])

                # Create new in-context data with permuted texts but original labels
                permuted_in_context_data = []
                for idx in range(len(in_context_data)):
                    example = in_context_data[idx].copy()
                    text_key = get_text_key(example)
                    example[text_key] = permuted_texts[idx]
                    permuted_in_context_data.append(example)

                results_path = f'{output_folder}/{model_name}/generated/{dataset_name}_set{set_run}_perm{perm_run}_detailed_results.csv'

                # Evaluate the model
                accuracy = 0
                if is_instruction_tuned:
                    accuracy, _, _ = evaluate_it_model(llm, dataset_name, test_data, permuted_in_context_data, create_prompt_fn, label_names, results_path=results_path)
                else:
                    accuracy, _, _ = evaluate_model(llm, dataset_name, test_data, permuted_in_context_data, create_prompt_fn, label_names, results_path=results_path)
                all_accuracies.append(accuracy)

                # Store the results by set, then permutation
                ordering_accuracies.setdefault(set_run, []).append(accuracy)

    # Prepare the CSV data: columns 0-9 for the first set, 10-19 for the second set, etc.
    flattened_accuracies = []
    for set_run in range(num_sets):
        flattened_accuracies.extend(ordering_accuracies[set_run])

    # Output the CSV with accuracies
    header_row = list(range(len(flattened_accuracies)))
    data_row = flattened_accuracies
    accuracy_df = pd.DataFrame([data_row], columns=header_row)
    accuracy_df.to_csv(f'{output_folder}/{model_name}/results/{dataset_name}_accuracy_results.csv', index=False)
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_accuracies)
    plt.title(f'Accuracy Distribution for {dataset_name}')
    plt.ylabel('Accuracy')
    plt.savefig(f'{output_folder}/{model_name}/boxplots/{dataset_name}_accuracy_boxplot.png')
    plt.close()

    # Create frequency graph (histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(all_accuracies, bins=20, edgecolor='black')
    plt.title(f'Accuracy Frequency for {dataset_name}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_folder}/{model_name}/frequency_graphs/{dataset_name}_accuracy_histogram.png')
    plt.close()

print("Experiments completed for all datasets.")
