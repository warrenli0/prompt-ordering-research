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
model_names = ["google/gemma-2b", "google/gemma-7b", "google/gemma-2-2b", "google/gemma-2-9b"]
token = "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"
dataset_list = ["gsm8k"]
output_folder = "outputs/train_ordering"
num_samples = 5000
num_test_examples = 500
num_runs_per_set = 10  # Evaluate each set 10 times
num_sets = 10  # Total of 10 different sets of in-context examples
multiples = [2]  # 2 examples per label
shuffle_seed = 42
label_names = None

# Configuration
train_set_size = 100 # Added for the selection of 50 or 100 examples

# Get the correct key for text/content
def get_text_key(example):
    if 'content' in example:
        return example['content']
    elif 'text' in example:
        return example['text']
    elif 'sentence' in example:
        return example['sentence']
    elif 'question' in example:
        return example['question']
    else:
        raise KeyError("Neither 'content' nor 'text' key is found in the example")

# Get the correct key for label
def get_label_key(example):
    if 'label' in example:
        return example['label']
    elif 'answer' in example:
        return example['answer']
    else:
        raise KeyError("Neither key is found in the example")

def evaluate(model_name):
    print(f"Running for {model_name}")
    is_instruction_tuned = model_name.endswith('it')

    # Setup
    llm = setup_model(model_name, token)

    for dataset_name in dataset_list:
        print(f"Running for {dataset_name}")

        label_key = 'answer' if dataset_name in ['mmlu', 'gsm8k'] else 'label'

        # Load dataset
        train_set, test_set, num_classes, label_names = load_dataset_by_name(dataset_name, num_samples)
        test_data = load_test_set(test_set, label_key, train_set_size, dataset_name)

        total_examples_per_class = num_classes * 5  # Store 5x more examples than needed
        sampled_data = prefilter_and_sample_examples_multiple(train_set, num_classes, total_examples_per_class, dataset_name, seed=shuffle_seed)

        # Get the correct prompt creation function
        create_prompt_fn = get_prompt_creator(dataset_name, is_instruction_tuned=is_instruction_tuned, label_names=label_names)

        # Create output directories
        output_dirs = [f'{output_folder}/{model_name}/results', f'{output_folder}/{model_name}/boxplots', f'{output_folder}/{model_name}/frequency_graphs']
        for dir_name in output_dirs:
            os.makedirs(dir_name, exist_ok=True)

        # Initialize containers for storing results
        all_accuracies = []
        ordering_accuracies = {}
        all_majority_accuracies = []
        best_ordering_accuracies = []
        best_orderings = []

        # Iterate over each multiple
        for multiple in multiples:
            num_incontext_examples = num_classes * multiple

            # Generate the random sets of in-context examples only once
            in_context_sets = []
            for set_run in range(num_sets):
                in_context_data = select_in_context_examples_multiple(sampled_data, num_classes, num_incontext_examples, dataset_name, seed=set_run + shuffle_seed)
                in_context_data = sorted(in_context_data, key=lambda x: (x[label_key], get_text_key(x)))
                in_context_sets.append(in_context_data)  # Store each set

            # Generate different sets of in-context examples
            for set_run in tqdm(range(num_sets), desc=f"Sets for {dataset_name} with train size {train_set_size}"):
                prompt_labels = {i: [] for i in range(train_set_size)}
                ordering_predictions = []  # Store predictions per ordering
                ordering_permutations = []  # Store permutations

                # In-context examples for this set
                in_context_data = in_context_sets[set_run]

                # Sort by label and within each label alphabetically by text
                in_context_data = sorted(in_context_data, key=lambda x: (x[label_key], get_text_key(x)))

                # Generate unique orderings per in-context example set
                orderings = generate_random_orderings([0] * num_incontext_examples, num_orderings=num_runs_per_set)

                # Evaluate this set using each of the permutations
                for perm_run in range(num_runs_per_set):
                    # Apply the permutation
                    permuted_in_context_data = reorder_list(in_context_data, orderings[perm_run])

                    results_path = f'{output_folder}/{model_name}/generated/{dataset_name}_set{set_run}_perm{perm_run}_detailed_results.csv'

                    # Evaluate the model
                    accuracy = 0
                    predicted_labels = []
                    if is_instruction_tuned:
                        accuracy, _, predicted_labels = evaluate_it_model(llm, dataset_name, test_data, permuted_in_context_data, create_prompt_fn, label_names, results_path=results_path)
                    else:
                        accuracy, _, predicted_labels = evaluate_model(llm, dataset_name, test_data, permuted_in_context_data, create_prompt_fn, label_names, results_path=results_path)
                    all_accuracies.append(accuracy)

                    # Collect predictions for this permutation
                    ordering_predictions.append(predicted_labels)
                    ordering_permutations.append(orderings[perm_run])

                    for idx, label in enumerate(predicted_labels):
                        prompt_labels[idx].append(label)

                    # Store the results by set, then permutation
                    ordering_accuracies.setdefault(set_run, []).append(accuracy)

                # Calculate majority vote label for each prompt and evaluate accuracy
                correct_majority_votes = 0
                majority_labels = []
                for idx in range(train_set_size):
                    majority_label = Counter(prompt_labels[idx]).most_common(1)[0][0]
                    majority_labels.append(majority_label)
                    true_label = str(test_data[idx][label_key])
                    if majority_label == true_label:
                        correct_majority_votes += 1

                # Store accuracy for this set
                majority_accuracy = correct_majority_votes / train_set_size
                all_majority_accuracies.append(majority_accuracy)

                # Compare each ordering's predictions with the majority vote predictions
                best_alignment = -1
                best_ordering_index = -1
                for ordering_index, predictions in enumerate(ordering_predictions):
                    alignment_count = sum([1 for idx in range(train_set_size) if predictions[idx] == majority_labels[idx]])
                    if alignment_count > best_alignment:
                        best_alignment = alignment_count
                        best_ordering_index = ordering_index

                # Retrieve the accuracy and permutation of the best ordering
                best_ordering_accuracy = ordering_accuracies[set_run][best_ordering_index]
                best_ordering_accuracies.append(best_ordering_accuracy)
                best_orderings.append(ordering_permutations[best_ordering_index])

            # Final evaluation with the top 10 orderings
            final_accuracies = []
            test_data = load_test_set(test_set, label_key, num_test_examples, dataset_name, seed=68)
            for set_run in range(num_sets):
                permuted_in_context_data = reorder_list(in_context_sets[set_run], best_orderings[set_run])
                accuracy, _, _ = evaluate_model(llm, dataset_name, test_data, permuted_in_context_data, create_prompt_fn, label_names)
                final_accuracies.append({
                    'Set': set_run,
                    'Best Ordering Train Accuracy': best_ordering_accuracies[set_run],
                    'Final Evaluation Accuracy': accuracy,
                    'Ordering': best_orderings[set_run]
                })

            # Save final results
            final_accuracies_df = pd.DataFrame(final_accuracies)
            final_accuracies_df.to_csv(f'{output_folder}/{model_name}/results/{dataset_name}_final_best_orderings_evaluation.csv', index=False)

for model_name in model_names:
    evaluate(model_name)

print("Experiments completed for all datasets and training set sizes.")
