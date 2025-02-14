import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from ast import literal_eval

#############################################
# Helper functions
#############################################

def combine_csv_files_with_gpu(pattern, local_runs=16):
    """
    Find and combine CSV files matching the given glob pattern.
    For each file, parse out the GPU index from the filename and
    add a new column 'global_perm' computed as:
         global_perm = permutation_number + (gpu_index * local_runs)
    """
    files = glob.glob(pattern)
    if not files:
        return None
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            m = re.search(r"gpu(\d+)\.csv", f)
            gpu_index = int(m.group(1)) if m else 0
            df['global_perm'] = df['permutation_number'] + gpu_index * local_runs
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return None

#############################################
# Process all_predictions CSVs
#############################################

def process_all_predictions(model, dataset, generated_path, local_runs=16):
    """
    For the given model and dataset (and its generated folder),
    combine all _all_predictions CSVs, compute per-run accuracies,
    then group by set_number to obtain:
      - average accuracy,
      - standard deviation,
      - majority vote accuracy,
      - best ordering accuracy.
      
    Also return a dict with the test accuracies keyed by set_number and global permutation.
    """
    pattern = os.path.join(generated_path, f"{dataset}_all_predictions_gpu*.csv")
    df = combine_csv_files_with_gpu(pattern, local_runs=local_runs)
    if df is None:
        return None, None
    
    # Convert string representations of lists into actual lists.
    df['predictions'] = df['predictions'].apply(literal_eval)
    df['correct_answers'] = df['correct_answers'].apply(literal_eval)
    
    # Compute accuracy per row (each row corresponds to one permutation run)
    def compute_accuracy(row):
        preds = row['predictions']
        correct = row['correct_answers']
        if len(preds) != len(correct):
            return np.nan
        return np.mean([str(p) == str(c) for p, c in zip(preds, correct)])
    
    df['accuracy'] = df.apply(compute_accuracy, axis=1)
    
    # Dictionary to hold test accuracies: {set_number: {global_perm: accuracy}}
    test_accuracies_by_set = defaultdict(dict)
    set_summary = []
    
    for set_num, group in df.groupby('set_number'):
        avg_acc = group['accuracy'].mean()
        std_acc = group['accuracy'].std(ddof=1)
        
        # Compute majority vote accuracy over test examples.
        n_examples = len(group.iloc[0]['predictions'])
        majority_preds = []
        for i in range(n_examples):
            preds_i = [row['predictions'][i] for _, row in group.iterrows()]
            maj = Counter(preds_i).most_common(1)[0][0]
            majority_preds.append(maj)
        # Assuming all rows in a set have the same correct answers.
        correct = group.iloc[0]['correct_answers']
        mv_acc = np.mean([str(maj) == str(c) for maj, c in zip(majority_preds, correct)])
        
        # Best ordering accuracy (maximum among the permutation runs)
        best_order_acc = group['accuracy'].max()
        
        # Save the individual test accuracies by global permutation.
        for _, row in group.iterrows():
            gperm = row['global_perm']
            test_accuracies_by_set[set_num][gperm] = row['accuracy']
        
        set_summary.append({
            'set_number': set_num,
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'majority_vote_accuracy': mv_acc,
            'best_ordering_accuracy': best_order_acc
        })
    
    set_df = pd.DataFrame(set_summary)
    
    overall_summary = {
        'overall_avg_accuracy': set_df['avg_accuracy'].mean(),
        'overall_std_accuracy': set_df['std_accuracy'].mean(),
        'avg_grouped_std': set_df['std_accuracy'].mean(),  # average of set std deviations
        'avg_majority_vote_accuracy': set_df['majority_vote_accuracy'].mean(),
        'avg_best_ordering_accuracy': set_df['best_ordering_accuracy'].mean()
    }
    
    return overall_summary, test_accuracies_by_set

#############################################
# Process train_predictions CSVs
#############################################

def process_train_predictions(model, dataset, generated_path, train_size=1000, local_runs=16):
    """
    For the given model and dataset, combine all train_predictions CSVs
    (assumed to be named as {dataset}_{train_size}_train_predictions_gpu*.csv),
    compute overall training accuracy, and for each set determine the
    permutation run whose predictions best align with the majority vote.
    
    Returns overall train accuracy, a dict mapping set_number to best global permutation,
    and a nested dict of training predictions per set and global permutation.
    """
    pattern = os.path.join(generated_path, f"{dataset}_{train_size}_train_predictions_gpu*.csv")
    df = combine_csv_files_with_gpu(pattern, local_runs=local_runs)
    if df is None:
        return None
    
    # Compute whether each training prediction is correct.
    df['train_acc'] = df.apply(lambda row: 1 if str(row['prediction']) == str(row['correct_answer']) else 0, axis=1)
    overall_train_acc = df['train_acc'].mean()
    
    # Group predictions by set_number and global permutation.
    train_groups = df.groupby(['set_number', 'global_perm'])
    train_preds = defaultdict(dict)
    for (set_num, gperm), group in train_groups:
        preds = group['prediction'].tolist()
        train_preds[set_num][gperm] = preds
    
    # For each set, compute the majority vote for each training example and then find
    # the permutation run with the highest alignment.
    best_perm_per_set = {}
    for set_num, perm_dict in train_preds.items():
        ex_count = len(next(iter(perm_dict.values())))
        majority_votes = []
        for i in range(ex_count):
            preds_i = [perm_dict[perm][i] for perm in perm_dict]
            maj = Counter(preds_i).most_common(1)[0][0]
            majority_votes.append(maj)
        
        best_align = -1
        best_perm = None
        for perm, preds in perm_dict.items():
            align = sum(1 for p, m in zip(preds, majority_votes) if str(p) == str(m))
            if align > best_align:
                best_align = align
                best_perm = perm
        best_perm_per_set[set_num] = best_perm
    
    return overall_train_acc, best_perm_per_set, train_preds

#############################################
# Subsample Train Orderings for Graphing
#############################################

def subsample_train_orderings(test_accuracies_by_set, train_preds, subsample_sizes):
    """
    For each subsample size k, for each set, choose k permutation runs (using a reproducible sorted order),
    compute three metrics:
       - 'best_alignment': the test accuracy of the ordering chosen via training alignment.
       - 'max': the maximum test accuracy among the sampled orderings.
       - 'mean': the average test accuracy among the sampled orderings.
    Returns a dict mapping subsample size k to a dict:
         { 'best_alignment': avg over sets,
           'max': avg over sets,
           'mean': avg over sets }
    """
    results = {}
    for k in subsample_sizes:
        best_alignment_list = []
        max_list = []
        mean_list = []
        for set_num, perm_dict in train_preds.items():
            perms = list(perm_dict.keys())
            if len(perms) < k:
                continue
            # For reproducibility, use the first k in sorted order.
            sampled_perms = sorted(perms)[:k]
            ex_count = len(perm_dict[sampled_perms[0]])
            # Compute majority vote for training predictions among the sampled orderings.
            majority_votes = []
            for i in range(ex_count):
                preds_i = [perm_dict[p][i] for p in sampled_perms]
                maj = Counter(preds_i).most_common(1)[0][0]
                majority_votes.append(maj)
            best_align = -1
            best_perm = None
            for p in sampled_perms:
                align = sum(1 for pred, maj in zip(perm_dict[p], majority_votes) if str(pred) == str(maj))
                if align > best_align:
                    best_align = align
                    best_perm = p
            # Get test accuracies for the sampled orderings.
            if set_num in test_accuracies_by_set:
                test_accs = [test_accuracies_by_set[set_num].get(p, np.nan) for p in sampled_perms]
                test_accs = [acc for acc in test_accs if not np.isnan(acc)]
                if test_accs:
                    best_alignment_value = test_accuracies_by_set[set_num].get(best_perm, np.nan)
                    max_value = max(test_accs)
                    mean_value = np.mean(test_accs)
                    best_alignment_list.append(best_alignment_value)
                    max_list.append(max_value)
                    mean_list.append(mean_value)
        if best_alignment_list:
            results[k] = {
                'best_alignment': np.mean(best_alignment_list),
                'max': np.mean(max_list),
                'mean': np.mean(mean_list)
            }
        else:
            results[k] = {'best_alignment': None, 'max': None, 'mean': None}
    return results

#############################################
# Main processing: loop over models and datasets
#############################################

def main():
    base_path = "outputs/combined"
    analysis_dir = "analysis_results/combined"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # These lists will hold summary rows for our Excel workbook.
    all_predictions_summary = []
    train_predictions_summary = []
    subsample_graph_data = []
    
    # Walk through the outputs/combined directory.
    # Expected structure: outputs/combined/<first_folder>/<second_folder>/generated
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) == "generated":
            parts = root.split(os.sep)
            try:
                # Construct model name from the two folders immediately under outputs/combined.
                model = os.path.join(parts[2], parts[3])
            except IndexError:
                print(f"Skipping folder {root} (unexpected structure)")
                continue
            
            # Determine dataset names by scanning the file names.
            datasets = set()
            for file in os.listdir(root):
                if file.endswith("_all_predictions_gpu0.csv"):
                    # File format: {dataset}_all_predictions_gpu{index}.csv
                    dataset_name = file.split("_all_predictions")[0]
                    datasets.add(dataset_name)
            
            for dataset in datasets:
                print(f"Processing model: {model}, dataset: {dataset}")
                # Process test predictions.
                all_pred_summary, test_accuracies_by_set = process_all_predictions(model, dataset, root, local_runs=16)
                if all_pred_summary is None:
                    print(f"  No all_predictions CSVs found for {model}, {dataset}")
                    continue
                # Process training predictions (assuming training set size = 1000).
                train_result = process_train_predictions(model, dataset, root, train_size=1000, local_runs=16)
                if train_result is None:
                    print(f"  No train_predictions CSVs found for {model}, {dataset}")
                    continue
                overall_train_acc, best_perm_per_set, train_preds = train_result
                
                # For each set, lookup the test accuracy for the best-aligned ordering.
                best_test_accs = []
                for set_num, best_perm in best_perm_per_set.items():
                    if set_num in test_accuracies_by_set and best_perm in test_accuracies_by_set[set_num]:
                        best_test_accs.append(test_accuracies_by_set[set_num][best_perm])
                avg_test_from_best_alignment = np.mean(best_test_accs) if best_test_accs else None
                
                # Save summary for train_predictions.
                train_predictions_summary.append({
                    "Model": model,
                    "Dataset": dataset,
                    "Overall_Train_Accuracy": overall_train_acc,
                    "Avg_Test_Accuracy_from_Best_Train_Alignment": avg_test_from_best_alignment
                })
                
                # Save summary for all_predictions.
                summary_entry = {"Model": model, "Dataset": dataset}
                summary_entry.update(all_pred_summary)
                all_predictions_summary.append(summary_entry)
                
                # For the subsample graph, use a list of subsample sizes.
                subsample_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
                subsample_results = subsample_train_orderings(test_accuracies_by_set, train_preds, subsample_sizes)
                for k, metrics in subsample_results.items():
                    subsample_graph_data.append({
                        "Model": model,
                        "Dataset": dataset,
                        "Train_Orderings": k,
                        "Best_Alignment": metrics['best_alignment'],
                        "Max": metrics['max'],
                        "Mean": metrics['mean']
                    })
    
    # Create DataFrames for summary tables.
    all_pred_df = pd.DataFrame(all_predictions_summary)
    train_pred_df = pd.DataFrame(train_predictions_summary)
    subsample_df = pd.DataFrame(subsample_graph_data)
    
    # Write summary tables to an Excel workbook.
    writer = pd.ExcelWriter(os.path.join(analysis_dir, "summary_results.xlsx"), engine='xlsxwriter')
    all_pred_df.to_excel(writer, sheet_name="All_Predictions", index=False)
    train_pred_df.to_excel(writer, sheet_name="Train_Predictions", index=False)
    subsample_df.to_excel(writer, sheet_name="Subsample_Graph_Data", index=False)
    writer.close()
    
    # Generate a plot for each model/dataset combination (three curves: Best Alignment, Max, Mean).
    for (model, dataset), group in subsample_df.groupby(["Model", "Dataset"]):
        plt.figure(figsize=(8, 6))
        plt.plot(group["Train_Orderings"], group["Best_Alignment"], marker="o", label="Best Alignment")
        plt.plot(group["Train_Orderings"], group["Max"], marker="o", label="Max Ordering Accuracy")
        plt.plot(group["Train_Orderings"], group["Mean"], marker="o", label="Average Ordering Accuracy")
        plt.xlabel("Number of Train Orderings")
        plt.ylabel("Test Accuracy")
        plt.title(f"Test Accuracy vs. Number of Train Orderings\nModel: {model}, Dataset: {dataset}")
        plt.legend()
        plt.grid(True)
        # Replace "/" in model names for filename compatibility.
        model_safe = model.replace("/", "_")
        plt.savefig(os.path.join(analysis_dir, f"{model_safe}_{dataset}_subsample_graph.png"))
        plt.close()
    
    # Also create an aggregate plot over all model/dataset combinations.
    plt.figure(figsize=(10, 8))
    for (model, dataset), group in subsample_df.groupby(["Model", "Dataset"]):
        label = f"{model}_{dataset}"
        plt.plot(group["Train_Orderings"], group["Best_Alignment"], marker="o", label=label)
    plt.xlabel("Number of Train Orderings")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs. Number of Train Orderings (Aggregate - Best Alignment)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(analysis_dir, "aggregate_subsample_graph.png"))
    plt.close()
    
    print("Analysis complete. Summary results and plots saved in", analysis_dir)

#############################################
# Entry point
#############################################

if __name__ == '__main__':
    main()
