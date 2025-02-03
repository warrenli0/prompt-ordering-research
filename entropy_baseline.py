import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Existing modules
from model_setup import setup_model
from data_loader import (
    load_dataset_by_name,
    prefilter_and_sample_examples_multiple,
    select_in_context_examples_multiple,
    load_test_set
)
from prompt_creators import get_prompt_creator
from evaluator import evaluate_model, evaluate_permutation_for_entropy
from utils import reorder_list, generate_random_orderings

output_folder = "outputs/entropy_baseline"

def generate_probing_set_from_training(train_data, size=20, seed=42):
    """
    Randomly selects 'size' examples from 'train_data' to form a probing set.
    'train_data' is a list of examples (dicts, etc.).
    """
    import random
    random.seed(seed)
    if len(train_data) <= size:
        return train_data
    return random.sample(train_data, size)


def compute_entropy_score(llm, permuted_incontext_data, create_prompt_fn, label_names, probing_set, metric="localE"):
    return evaluate_permutation_for_entropy(
        llm=llm,
        in_context_examples=permuted_incontext_data,
        probing_set=probing_set,
        create_prompt_fn=create_prompt_fn,
        label_names=label_names,
        temperature=2.0,
        max_tokens=128,
        block_ngram_repeat=2,
        metric=metric
    )

def choose_performant_prompts(llm, in_context_data, create_prompt_fn, label_names, k=4, num_permutations=24, metric="localE"):
    """
    1) Generate candidate permutations
    2) For each permutation, compute either LocalE or GlobalE on a probing set
    3) Rank by score (descending)
    4) Return the top k permutations
    """
    permutations = generate_random_orderings([0]*len(in_context_data), num_orderings=num_permutations)
    
    # Build / load your probing set
    probing_set = generate_probing_set_from_training(in_context_data)

    scored_perms = []
    for i, perm in enumerate(permutations):
        permuted_incontext_data = reorder_list(in_context_data, perm)
        score = compute_entropy_score(llm, permuted_incontext_data, create_prompt_fn, label_names, probing_set, metric)
        scored_perms.append((i, score))

    # Sort desc by the metric
    ranked = sorted(scored_perms, key=lambda x: x[1], reverse=True)

    top_indices = [p[0] for p in ranked[:k]]
    top_permutations = [permutations[idx] for idx in top_indices]
    return top_permutations


model_names = ["google/gemma-2b", "google/gemma-7b", "google/gemma-2-2b", "google/gemma-2-9b"]
dataset_list = ["nyt-topics", "nyt-locations", "ag_news", "dbpedia"] # sst2, mmlu?
token = "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"

def evaluate(model_name):
    print(f"Running for {model_name}")
    is_instruction_tuned = model_name.endswith('it')

    # Setup model
    llm = setup_model(model_name, token)

    # Evaluate each dataset
    for dataset_name in dataset_list:
        print(f"Running for {dataset_name}")

        label_key = 'answer' if dataset_name in ['mmlu', 'gsm8k'] else 'label'
        train_set, test_set, num_classes, label_names = load_dataset_by_name(dataset_name, 5000)
        test_data = load_test_set(test_set, label_key, 500, dataset_name)

        total_examples_per_class = num_classes * 5
        sampled_data = prefilter_and_sample_examples_multiple(train_set, num_classes, total_examples_per_class, dataset_name, seed=42)

        create_prompt_fn = get_prompt_creator(dataset_name, is_instruction_tuned, label_names=label_names)

         # Create output directories
        output_dirs = [f'{output_folder}/{model_name}/results', 
                       f'{output_folder}/{model_name}/boxplots', 
                       f'{output_folder}/{model_name}/frequency_graphs',
                       f'{output_folder}/{model_name}/generated']
        for dir_name in output_dirs:
            os.makedirs(dir_name, exist_ok=True)

        multiples = [2]  
        num_sets = 10

        all_accuracies = []
        ordering_accuracies = {}

        for multiple in multiples:
            num_incontext_examples = num_classes * multiple
            in_context_sets = []
            for set_run in range(num_sets):
                ics = select_in_context_examples_multiple(sampled_data, num_classes, num_incontext_examples, dataset_name, seed=set_run+42)
                ics = sorted(ics, key=lambda x: (x[label_key], x['text'] if 'text' in x else x['content']))
                in_context_sets.append(ics)

            for set_run in tqdm(range(num_sets), desc=f"{dataset_name} sets"):
                in_context_data = in_context_sets[set_run]

                # Pick top permutations by LocalE (or GlobalE)
                top_permutations = choose_performant_prompts(
                    llm, 
                    in_context_data, 
                    create_prompt_fn, 
                    label_names=label_names, 
                    k=4, 
                    num_permutations=24, 
                    metric="localE"  # or "globalE"
                )

                for perm_idx, perm in enumerate(top_permutations):
                    permuted_data = reorder_list(in_context_data, perm)
                    results_path = f'{output_folder}/{model_name}/generated/{dataset_name}_set{set_run}_perm{perm_idx}_detailed.csv'

                    accuracy, _, _ = evaluate_model(
                        llm,
                        dataset_name,
                        test_data,
                        permuted_data,
                        create_prompt_fn,
                        label_names=label_names,
                        results_path=results_path
                    )
                    all_accuracies.append(accuracy)
                    ordering_accuracies.setdefault(set_run, []).append(accuracy)

        # Save CSV results
        flattened_accuracies = []
        for set_run in range(num_sets):
            flattened_accuracies.extend(ordering_accuracies.get(set_run, []))

        header_row = list(range(len(flattened_accuracies)))
        data_row = flattened_accuracies
        accuracy_df = pd.DataFrame([data_row], columns=header_row)
        accuracy_df.to_csv(
            f'{output_folder}/{model_name}/results/{dataset_name}_accuracy_results.csv',
            index=False
        )

        # Boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(all_accuracies)
        plt.title(f'Accuracy Distribution for {dataset_name}')
        plt.ylabel('Accuracy')
        plt.savefig(f'{output_folder}/{model_name}/boxplots/{dataset_name}_accuracy_boxplot.png')
        plt.close()

        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(all_accuracies, bins=20, edgecolor='black')
        plt.title(f'Accuracy Frequency for {dataset_name}')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.savefig(f'{output_folder}/{model_name}/frequency_graphs/{dataset_name}_accuracy_histogram.png')
        plt.close()


for mn in model_names:
    evaluate(mn)

print("Experiments completed for all datasets.")
