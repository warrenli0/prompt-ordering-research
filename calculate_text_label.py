import os
import glob
import pandas as pd
from collections import defaultdict

def calculate_grouped_stats(data, group_size=10, pattern_grouping=False):
    grouped_means = []
    grouped_stds = []
    
    if pattern_grouping:
        for i in range(group_size):
            pattern_columns = data.iloc[:, i::group_size]
            group_mean = pattern_columns.mean(axis=1)
            group_std = pattern_columns.std(axis=1)
            grouped_means.append(group_mean)
            grouped_stds.append(group_std)
    else:
        for i in range(0, data.shape[1], group_size):
            grouped_data = data.iloc[:, i:i + group_size]
            group_mean = grouped_data.mean(axis=1)
            group_std = grouped_data.std(axis=1)
            grouped_means.append(group_mean)
            grouped_stds.append(group_std)
    
    return grouped_means, grouped_stds

def calculate_stats_for_csv(file_path, pattern_grouping=False):
    data = pd.read_csv(file_path, header=None)
    data_without_header = data.iloc[1:]
    
    grouped_means, grouped_stds = calculate_grouped_stats(data_without_header, pattern_grouping=pattern_grouping)
    
    overall_mean = data_without_header.mean(axis=1).mean()
    overall_std = data_without_header.std(axis=1).mean()
    avg_grouped_std = pd.concat(grouped_stds, axis=1).mean(axis=1).mean()

    return overall_mean, overall_std, avg_grouped_std

def process_csv_files_in_subfolder(subfolder):
    dataset_results = defaultdict(lambda: {'mean': [], 'std': [], 'grouped_std': []})
    csv_files = glob.glob(os.path.join(subfolder, '*.csv'))
    
    for file_path in csv_files:
        dataset_name = os.path.basename(file_path).split('.')[0]
        overall_mean, overall_std, avg_grouped_std = calculate_stats_for_csv(file_path)
        
        dataset_results[dataset_name]['mean'].append(overall_mean)
        dataset_results[dataset_name]['std'].append(overall_std)
        dataset_results[dataset_name]['grouped_std'].append(avg_grouped_std)

    return dataset_results

def summarize_dataset_results(dataset_results):
    summary = {}
    for dataset, stats in dataset_results.items():
        if stats['mean']:
            summary[dataset] = {
                'mean': sum(stats['mean']) / len(stats['mean']),
                'std': sum(stats['std']) / len(stats['std']),
                'grouped_std': sum(stats['grouped_std']) / len(stats['grouped_std']),
            }
    return summary

def calculate_averages_across_datasets(dataset_summary):
    all_means = [stats['mean'] for stats in dataset_summary.values()]
    all_stds = [stats['std'] for stats in dataset_summary.values()]
    all_grouped_stds = [stats['grouped_std'] for stats in dataset_summary.values()]

    if all_means:
        return {
            'mean': sum(all_means) / len(all_means),
            'std': sum(all_stds) / len(all_stds),
            'grouped_std': sum(all_grouped_stds) / len(all_grouped_stds),
        }
    else:
        return {
            'mean': None,
            'std': None,
            'grouped_std': None,
        }
    
def summarize_aggregated_dataset_results(aggregated_dataset_results):
    summary = {}
    for dataset, stats in aggregated_dataset_results.items():
        if stats['mean']:
            summary[dataset] = {
                'mean': sum(stats['mean']) / len(stats['mean']),
                'std': sum(stats['std']) / len(stats['std']),
                'grouped_std': sum(stats['grouped_std']) / len(stats['grouped_std']),
            }
    return summary

def process_all_csv_files(base_directory, pattern_grouping=False):
    subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) 
                      if os.path.isdir(os.path.join(base_directory, d))]
    
    summary_file_path = os.path.join(base_directory, 'summary_results.txt')
    overall_means = []
    overall_stds = []
    overall_grouped_stds = []
    aggregated_dataset_results = defaultdict(lambda: {'mean': [], 'std': [], 'grouped_std': []})
    
    with open(summary_file_path, 'w') as summary_file:
        summary_file.write("Summary of results grouped by model and dataset:\n\n")
        
        for subdir in subdirectories:
            model_name = os.path.basename(subdir)
            results_dir = os.path.join(subdir, 'results')
            if not os.path.exists(results_dir):
                print(f"No results folder found in {subdir}")
                continue
            
            summary_file.write(f"Model: {model_name}\n\n")
            dataset_results = process_csv_files_in_subfolder(results_dir)
            dataset_summary = summarize_dataset_results(dataset_results)

            # Append model-level statistics to the overall dataset-level aggregation
            for dataset, stats in dataset_results.items():
                aggregated_dataset_results[dataset]['mean'].extend(stats['mean'])
                aggregated_dataset_results[dataset]['std'].extend(stats['std'])
                aggregated_dataset_results[dataset]['grouped_std'].extend(stats['grouped_std'])
    
            summary_file.write(f"  Results:\n")
            for dataset, stats in dataset_summary.items():
                summary_file.write(f"    Dataset: {dataset}\n")
                summary_file.write(f"      Mean: {stats['mean']}\n")
                summary_file.write(f"      Std Dev: {stats['std']}\n")
                summary_file.write(f"      Avg Grouped Std Dev: {stats['grouped_std']}\n")
                
            # Calculate model-level averages across datasets
            model_averages = calculate_averages_across_datasets(dataset_summary)
            if model_averages['mean'] is not None:
                summary_file.write(f"  Average Across Datasets:\n")
                summary_file.write(f"    Mean: {model_averages['mean']}\n")
                summary_file.write(f"    Std Dev: {model_averages['std']}\n")
                summary_file.write(f"    Avg Grouped Std Dev: {model_averages['grouped_std']}\n\n")
                
                overall_means.append(model_averages['mean'])
                overall_stds.append(model_averages['std'])
                overall_grouped_stds.append(model_averages['grouped_std'])
            
            summary_file.write("--- END MODEL ---\n\n")

        # Summarize aggregated results across all models for each dataset
        aggregated_summary = summarize_aggregated_dataset_results(aggregated_dataset_results)

        # Write the aggregated results to the summary file
        summary_file.write("Aggregated Summary of Results by Dataset Across All Models:\n\n")
        
        for dataset, stats in aggregated_summary.items():
            summary_file.write(f"Dataset: {dataset}\n")
            summary_file.write(f"  Mean: {stats['mean']}\n")
            summary_file.write(f"  Std Dev: {stats['std']}\n")
            summary_file.write(f"  Avg Grouped Std Dev: {stats['grouped_std']}\n\n")

        # Calculate overall statistics across all models and datasets
        summary_file.write("Overall Average Statistics Across All Models:\n\n")
        if overall_means:
            avg_mean = sum(overall_means) / len(overall_means)
            avg_std = sum(overall_stds) / len(overall_stds)
            avg_grouped_std = sum(overall_grouped_stds) / len(overall_grouped_stds)
            
            summary_file.write(f"Average Accuracy (Mean): {avg_mean}\n")
            summary_file.write(f"Average Standard Deviation: {avg_std}\n")
            summary_file.write(f"Average of Grouped Standard Deviations: {avg_grouped_std}\n\n")

if __name__ == "__main__":
    base_directory = '/data/wyl003/prompt-ordering-research/outputs/text_label_separation/gpt2-xl'
    pattern_grouping = False
    process_all_csv_files(base_directory, pattern_grouping=pattern_grouping)
