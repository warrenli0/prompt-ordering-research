import os
import pandas as pd
import numpy as np
from model_setup import setup_model
from data_loader import load_dataset_by_name, load_text_file, select_in_context_examples
from prompt_creators import create_prompt_gsm8k, create_prompt_bbh, create_prompt_mmlu, create_prompt_agnews, create_prompt_sst2, create_prompt_dbpedia, create_prompt_custom
from evaluator import evaluate_model
from utils import shuffle_within_label, randomize_label_order, reorder_list

# Configuration
model_name = "google/gemma-2b"  # Use gemma2b more
token = "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"

# Dataset configuration
dataset_name = "dbpedia"  # "gsm8k", "lukaemon/bbh", "cais/mmlu", "ag_news", "sst2", or "dbpedia"
num_samples = 500
num_runs = 5
multiples = [2]  # List of multiples of the number of classes for in-context examples
num_test_examples = 200
num_orderings = 10
shuffle_seed = 42  # Seed for shuffling

# Setup
llm = setup_model(model_name, token)

# Load test set once
train_set, test_set, num_classes = load_dataset_by_name(dataset_name, num_samples)
test_set = test_set.shuffle(seed=42)
test_data = test_set.select(range(num_test_examples))

# Output paths
results_dir = os.path.join('outputs', 'results')
generated_text_dir = os.path.join('outputs', 'generated_text')
tables_dir = os.path.join('outputs', 'tables')
summary_dir = os.path.join('outputs', 'summary')  # New directory for summary files
ordering_variance_dir = os.path.join('outputs', 'variance_comparison')  # Directory for new table
os.makedirs(results_dir, exist_ok=True)
os.makedirs(generated_text_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)  # Create the summary directory
os.makedirs(ordering_variance_dir, exist_ok=True)  # Create the ordering variance directory

# Load label names if using custom datasets
label_names = None
if dataset_name in ['nyt-topics', 'nyt-locations']:
    base_path = os.path.join('data', dataset_name.replace('nyt', 'NYT'))
    label_names = load_text_file(os.path.join(base_path, 'label_names.txt'))

# Iterate over each multiple
for multiple in multiples:
    # Calculate the number of in-context examples for the current multiple
    num_incontext_examples = num_classes * multiple

    # Initialize results containers for this multiple
    all_run_accuracies = []
    all_run_stddevs = []
    accuracy_table = np.zeros((num_runs, num_orderings))

    # Evaluate for each run
    for run in range(num_runs):
        train_set, _, _ = load_dataset_by_name(dataset_name, num_samples)
        train_set = train_set.shuffle(seed=run)
        in_context_data = select_in_context_examples(train_set, num_classes, num_incontext_examples)

        # Sort data by label and within each label, sort alphabetically by text
        in_context_data = sorted(in_context_data, key=lambda x: (x['label'], x['text']))

        # Shuffle within each label (deterministic using a seed)
        in_context_data = shuffle_within_label(in_context_data, seed=shuffle_seed)

        # Optionally shuffle the order of labels (deterministic using a seed)
        in_context_data = randomize_label_order(in_context_data, seed=shuffle_seed)

        # Generate the orderings using the shuffled data
        orderings = [list(range(len(in_context_data)))] * num_orderings
        for i in range(num_orderings):
            shuffled_data = shuffle_within_label(in_context_data, seed=shuffle_seed + i)
            orderings[i] = list(range(len(shuffled_data)))

        run_results = []
        run_correctness_dict = {}

        results_file = os.path.join(results_dir, f"{dataset_name}_evaluation_results_run_{run+1}_incontext_{num_incontext_examples}.txt")
        outputs_file = os.path.join(generated_text_dir, f"{dataset_name}_outputs_run_{run+1}_incontext_{num_incontext_examples}.txt")

        with open(results_file, "w") as f, open(outputs_file, "w") as output_file:
            f.write(f"Model Name: {model_name}\n")  # Include model name in the results file
            output_file.write(f"Model Name: {model_name}\n")  # Include model name in the outputs file

            for i, ordering in enumerate(orderings):
                ordered_examples = reorder_list(in_context_data, ordering)
                if dataset_name == 'gsm8k':
                    create_prompt_fn = create_prompt_gsm8k
                elif dataset_name == 'bbh':
                    create_prompt_fn = create_prompt_bbh
                elif dataset_name == 'mmlu':
                    create_prompt_fn = create_prompt_mmlu
                elif dataset_name == 'ag_news':
                    create_prompt_fn = create_prompt_agnews
                elif dataset_name == 'sst2':
                    create_prompt_fn = create_prompt_sst2
                elif dataset_name == 'dbpedia':
                    create_prompt_fn = create_prompt_dbpedia
                elif dataset_name in ['nyt-topics', 'nyt-locations']:
                    create_prompt_fn = create_prompt_custom

                output_file.write(f"Ordering {i}:\n")
                accuracy, correctness = evaluate_model(llm, dataset_name, test_data, ordered_examples, create_prompt_fn, label_names, output_file=output_file)
                run_results.append(accuracy)
                run_correctness_dict[i] = correctness
                accuracy_table[run, i] = accuracy  # Store accuracy in the table
                f.write(f"Ordering {i+1}: Accuracy = {accuracy}\n")
                f.write(f"Shuffled Order: {ordering}\n\n")
                print(f"Run {run+1}, Ordering {i+1}: Accuracy = {accuracy}")

        average_run_accuracy = np.mean(run_results)
        stddev_run_accuracy = np.std(run_results)
        all_run_accuracies.append(average_run_accuracy)
        all_run_stddevs.append(stddev_run_accuracy)

    # Compute row-wise and column-wise averages and standard deviations
    row_wise_avg = np.mean(accuracy_table, axis=1)
    row_wise_std = np.std(accuracy_table, axis=1)
    col_wise_avg = np.mean(accuracy_table, axis=0)
    col_wise_std = np.std(accuracy_table, axis=0)

    # Write the accuracy table with row-wise and column-wise statistics
    table_file = os.path.join(ordering_variance_dir, f"{dataset_name}_accuracy_table_incontext_{num_incontext_examples}.csv")
    table_summary_file = os.path.join(ordering_variance_dir, f"{dataset_name}_accuracy_summary_incontext_{num_incontext_examples}.txt")

    accuracy_df = pd.DataFrame(accuracy_table, columns=[f"Ordering {i+1}" for i in range(num_orderings)])
    accuracy_df['Row-wise Avg'] = row_wise_avg
    accuracy_df['Row-wise Std'] = row_wise_std
    accuracy_df.loc['Column-wise Avg'] = np.append(col_wise_avg, [np.nan, np.nan])
    accuracy_df.loc['Column-wise Std'] = np.append(col_wise_std, [np.nan, np.nan])

    accuracy_df.to_csv(table_file, index_label='Run')

    # Write the row-wise and column-wise averages and standard deviations to a summary file
    with open(table_summary_file, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Multiple: {multiple}\n")
        f.write("Row-wise Averages and Standard Deviations (Run-wise comparison):\n")
        for i, (avg, std) in enumerate(zip(row_wise_avg, row_wise_std)):
            f.write(f"Run {i+1}: Average = {avg}, Std Dev = {std}\n")

        f.write("\nColumn-wise Averages and Standard Deviations (Ordering-wise comparison):\n")
        for i, (avg, std) in enumerate(zip(col_wise_avg, col_wise_std)):
            f.write(f"Ordering {i+1}: Average = {avg}, Std Dev = {std}\n")

    # Write average accuracy and standard deviation of each run to a summary file
    summary_file = os.path.join(summary_dir, f"{dataset_name}_summary_incontext_{num_incontext_examples}.txt")
    with open(summary_file, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Multiple: {multiple}\n")
        for run, (avg_accuracy, stddev_accuracy) in enumerate(zip(all_run_accuracies, all_run_stddevs), start=1):
            f.write(f"Run {run}: Average Accuracy = {avg_accuracy}, Std Dev = {stddev_accuracy}\n")

    # Save detailed correctness results
    for run in range(num_runs):
        correctness_df = pd.DataFrame(run_correctness_dict)
        correctness_df.to_csv(os.path.join(tables_dir, f"{dataset_name}_correctness_comparison_run_{run+1}_incontext_{num_incontext_examples}.csv"), index=False)
