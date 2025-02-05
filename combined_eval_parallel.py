import os
import argparse
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

# Parse command-line arguments for GPU splitting
parser = argparse.ArgumentParser(description="Run experiments on a specified GPU shard.")
parser.add_argument('--gpu_index', type=int, default=0, help='Index of this GPU (0-indexed)')
parser.add_argument('--num_gpus', type=int, default=8, help='Total number of GPUs available')
args = parser.parse_args()
gpu_index = args.gpu_index
num_gpus = args.num_gpus

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

# Configuration
model_names = [
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
]
token = "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"
dataset_list = ["nyt-topics", "nyt-locations", "dbpedia", "gsm8k"]  # Example datasets
output_folder = "outputs/combined"
num_samples = 5000
num_test_examples = 500
train_ordering_train_sizes = [1000]  # Training set sizes for the train_ordering experiment
num_runs_per_set = 128  # Total number of runs per set (to be split among GPUs)
num_sets = 10           # Total of 10 different sets of in-context examples
multiples = [2]         # 2 examples per label
shuffle_seed = 42
label_names = None

# Compute how many runs this GPU will perform per set
local_runs = num_runs_per_set // num_gpus  # For example, 128/8 = 16

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
    print(f"Running for {model_name} on GPU {gpu_index}")
    is_instruction_tuned = model_name.endswith('it')
    llm = setup_model(model_name, token)

    for dataset_name in dataset_list:
        print(f"Running for {dataset_name}")
        # Choose the key for the label based on the dataset
        label_key = 'answer' if dataset_name in ['mmlu', 'gsm8k'] else 'label'

        # Load datasets
        train_set, test_set, num_classes, label_names = load_dataset_by_name(dataset_name, num_samples)
        test_data = load_test_set(test_set, label_key, num_test_examples, dataset_name)

        # Get correct answers for test set
        correct_answers = []
        for example in test_data:
            answer = example[label_key]
            if dataset_name == 'gsm8k':
                answer = answer.split('####')[-1].strip()
            correct_answers.append(str(answer))

        # Load training data for train_ordering experiment
        train_ordering_datasets = {}
        for size in train_ordering_train_sizes:
            # Use load_test_set to ensure a balanced set and use a seed based on the size
            shuffled_train = load_test_set(train_set, label_key, size, dataset_name, seed=shuffle_seed + size)
            train_ordering_datasets[size] = shuffled_train

        # Prefilter examples for in-context selection
        total_examples_per_class = num_classes * 5
        sampled_data = prefilter_and_sample_examples_multiple(train_set, num_classes, total_examples_per_class, dataset_name, seed=shuffle_seed)

        create_prompt_fn = get_prompt_creator(dataset_name, is_instruction_tuned=is_instruction_tuned, label_names=label_names)

        # Create output directories
        output_base = f'{output_folder}/{model_name}'
        output_dirs = [
            f'{output_base}/results/example_to_order',
            f'{output_base}/results/majority_vote',
            f'{output_base}/results/best_ordering',
            f'{output_base}/results/train_ordering',
            f'{output_base}/boxplots',
            f'{output_base}/frequency_graphs',
            f'{output_base}/generated'
        ]
        for dir_name in output_dirs:
            os.makedirs(dir_name, exist_ok=True)

        # Initialize result containers (these will hold results only for this GPU’s local runs)
        example_to_order_accuracies = []
        majority_vote_accuracies = []
        best_ordering_accuracies = []
        train_ordering_accuracies = {size: [] for size in train_ordering_train_sizes}
        all_predictions = []
        all_train_predictions = []

        for multiple in multiples:
            num_incontext = num_classes * multiple

            # Pre-generate in-context example sets
            in_context_sets = []
            for set_run in range(num_sets):
                in_context_data = select_in_context_examples_multiple(
                    sampled_data, num_classes, num_incontext, dataset_name, seed=set_run + shuffle_seed
                )
                # Sort examples for consistency
                in_context_data = sorted(in_context_data, key=lambda x: (x[label_key], get_text_key(x)))
                in_context_sets.append(in_context_data)

            for set_run in tqdm(range(num_sets), desc=f"Processing sets for {dataset_name}"):
                in_context_data = in_context_sets[set_run]

                # Generate the full list of orderings (global orderings)
                global_orderings = generate_random_orderings([0] * num_incontext, num_orderings=num_runs_per_set, seed=set_run + shuffle_seed)
                # Select the subset corresponding to this GPU
                local_orderings = global_orderings[gpu_index * local_runs: (gpu_index + 1) * local_runs]

                # Collectors for this set (local runs only)
                test_accs = []
                test_preds = []
                test_prompt_labels = {i: [] for i in range(num_test_examples)}
                train_prompt_labels = {size: {i: [] for i in range(size)} for size in train_ordering_train_sizes}
                train_preds = {size: [] for size in train_ordering_train_sizes}

                # Iterate only over this GPU's chunk of runs
                for perm_run, ordering in enumerate(local_orderings):
                    permuted_in_context = reorder_list(in_context_data, ordering)

                    # Evaluate on main test set
                    if is_instruction_tuned:
                        test_acc, _, test_labels = evaluate_it_model(
                            llm, dataset_name, test_data, permuted_in_context, create_prompt_fn, label_names
                        )
                    else:
                        test_acc, _, test_labels = evaluate_model(
                            llm, dataset_name, test_data, permuted_in_context, create_prompt_fn, label_names
                        )
                    test_accs.append(test_acc)
                    test_preds.append(test_labels)
                    for idx, lbl in enumerate(test_labels):
                        test_prompt_labels[idx].append(lbl)

                    # Save prediction details for this permutation run
                    prediction_entry = {
                        'model': model_name,
                        'dataset': dataset_name,
                        'set_number': set_run,
                        'permutation_number': perm_run,
                        'ordering': ordering,
                        'predictions': test_labels,
                        'correct_answers': correct_answers
                    }
                    all_predictions.append(prediction_entry)

                    # Evaluate on train_ordering training sets for each size
                    current_train_preds = {}
                    for size in train_ordering_train_sizes:
                        train_data = train_ordering_datasets[size]
                        if is_instruction_tuned:
                            _, _, train_labels = evaluate_it_model(
                                llm, dataset_name, train_data, permuted_in_context, create_prompt_fn, label_names
                            )
                        else:
                            _, _, train_labels = evaluate_model(
                                llm, dataset_name, train_data, permuted_in_context, create_prompt_fn, label_names
                            )
                        current_train_preds[size] = train_labels
                        for idx in range(size):
                            train_prompt_labels[size][idx].append(train_labels[idx])
                        
                        # Save training prediction details
                        for idx, example in enumerate(train_data):
                            answer = example[label_key]
                            if dataset_name == 'gsm8k':
                                answer = answer.split('####')[-1].strip()
                            train_pred_entry = {
                                'model': model_name,
                                'dataset': dataset_name,
                                'set_number': set_run,
                                'permutation_number': perm_run,
                                'ordering': ordering,
                                'training_set_size': size,
                                'prediction': train_labels[idx],
                                'correct_answer': str(answer)
                            }
                            all_train_predictions.append(train_pred_entry)
                    for size in train_ordering_train_sizes:
                        train_preds[size].append(current_train_preds[size])

                # Now compute aggregate metrics for this set based on the local runs only
                # (When combining results from all GPUs later you may want to recompute these on the combined data.)
                mv_correct = 0
                for idx in range(num_test_examples):
                    majority_lbl = Counter(test_prompt_labels[idx]).most_common(1)[0][0]
                    answer = test_data[idx][label_key]
                    if dataset_name == 'gsm8k':
                        answer = test_data[idx][label_key].split('####')[-1].strip()
                    if majority_lbl == str(answer):
                        mv_correct += 1
                majority_vote_acc = mv_correct / num_test_examples
                majority_vote_accuracies.append(majority_vote_acc)

                best_test_idx = np.argmax(test_accs)
                best_ordering_accuracies.append(test_accs[best_test_idx])

                for size in train_ordering_train_sizes:
                    # Compute majority vote on the training set predictions
                    train_majority = [
                        Counter(train_prompt_labels[size][idx]).most_common(1)[0][0]
                        for idx in range(size)
                    ]
                    best_alignment = -1
                    best_train_idx = 0
                    # Note: iterate over local_runs (not the full num_runs_per_set)
                    for i in range(local_runs):
                        preds = train_preds[size][i]
                        alignment = sum(p == m for p, m in zip(preds, train_majority))
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_train_idx = i
                    train_ordering_accuracies[size].append(test_accs[best_train_idx])

                example_to_order_accuracies.extend(test_accs)
            
        # Save predictions after processing all permutations for this set, with GPU identifier appended
        prediction_df = pd.DataFrame(all_predictions)
        prediction_df.to_csv(
            f'{output_base}/generated/{dataset_name}_all_predictions_gpu{gpu_index}.csv',
            index=False
        )

        # Save training predictions split by size, with GPU identifier appended
        train_pred_df = pd.DataFrame(all_train_predictions)
        for size in train_ordering_train_sizes:
            size_df = train_pred_df[train_pred_df['training_set_size'] == size]
            size_filename = f'{dataset_name}_{size}_train_predictions_gpu{gpu_index}.csv'
            size_df.to_csv(
                f'{output_base}/generated/{size_filename}',
                index=False
            )

        # Save all (local) results. (Later, you can combine results across GPUs.)
        results = {
            'example_to_order': example_to_order_accuracies,
            'majority_vote': majority_vote_accuracies,
            'best_ordering': best_ordering_accuracies,
            'train_ordering': train_ordering_accuracies
        }

        for exp_name, accs in results.items():
            if exp_name == 'train_ordering':
                for size, size_accs in accs.items():
                    df = pd.DataFrame({'Accuracy': size_accs})
                    size_filename = f'{dataset_name}_{size}_results_gpu{gpu_index}.csv'
                    df.to_csv(f'{output_base}/results/{exp_name}/{size_filename}', index=False)
                    
                    plt.figure(figsize=(10,6))
                    plt.boxplot(size_accs)
                    plt.title(f'Train Ordering Accuracy (Size {size}) - {dataset_name} [GPU {gpu_index}]')
                    plt.ylabel('Accuracy')
                    plt.savefig(f'{output_base}/boxplots/{dataset_name}_train_ordering_{size}_boxplot_gpu{gpu_index}.png')
                    plt.close()

                    plt.figure(figsize=(10,6))
                    plt.hist(size_accs, bins=20, edgecolor='black')
                    plt.title(f'Train Ordering Accuracy Frequency (Size {size}) - {dataset_name} [GPU {gpu_index}]')
                    plt.xlabel('Accuracy')
                    plt.ylabel('Frequency')
                    plt.savefig(f'{output_base}/frequency_graphs/{dataset_name}_train_ordering_{size}_histogram_gpu{gpu_index}.png')
                    plt.close()
            else:
                df = pd.DataFrame({'Accuracy': accs})
                df.to_csv(f'{output_base}/results/{exp_name}/{dataset_name}_results_gpu{gpu_index}.csv', index=False)
                
                plt.figure(figsize=(10,6))
                plt.boxplot(accs)
                plt.title(f'{exp_name} Accuracy Distribution - {dataset_name} [GPU {gpu_index}]')
                plt.ylabel('Accuracy')
                plt.savefig(f'{output_base}/boxplots/{dataset_name}_{exp_name}_boxplot_gpu{gpu_index}.png')
                plt.close()

                plt.figure(figsize=(10,6))
                plt.hist(accs, bins=20, edgecolor='black')
                plt.title(f'{exp_name} Accuracy Frequency - {dataset_name} [GPU {gpu_index}]')
                plt.xlabel('Accuracy')
                plt.ylabel('Frequency')
                plt.savefig(f'{output_base}/frequency_graphs/{dataset_name}_{exp_name}_histogram_gpu{gpu_index}.png')
                plt.close()

for model_name in model_names:
    evaluate(model_name)

print("All experiments completed successfully on GPU", gpu_index)
