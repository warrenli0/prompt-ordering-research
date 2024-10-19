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

def process_all_csv_files(base_directory, pattern_grouping=False):
    subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) 
                      if os.path.isdir(os.path.join(base_directory, d))]
    
    overall_results = defaultdict(lambda: {'example_to_order': {}, 'order_to_example': {}})
    summary_file_path = os.path.join(base_directory, 'summary_results.txt')
    overall_means = {'example_to_order': [], 'order_to_example': []}
    overall_stds = {'example_to_order': [], 'order_to_example': []}
    overall_grouped_stds = {'example_to_order': [], 'order_to_example': []}
    
    with open(summary_file_path, 'w') as summary_file:
        summary_file.write("Summary of results grouped by model and dataset:\n\n")
        
        for subdir in subdirectories:
            model_name = os.path.basename(subdir)
            summary_file.write(f"Model: {model_name}\n\n")
            results_dir = os.path.join(subdir, 'results')
            if not os.path.exists(results_dir):
                print(f"No results folder found in {subdir}")
                continue
            
            model_averages = {}

            for order_type in ['example_to_order', 'order_to_example']:
                folder_path = os.path.join(results_dir, order_type)
                dataset_results = process_csv_files_in_subfolder(folder_path)
                dataset_summary = summarize_dataset_results(dataset_results)
                
                model_averages[order_type] = calculate_averages_across_datasets(dataset_summary)
                overall_results[model_name][order_type] = dataset_summary
                
                summary_file.write(f"  {order_type}:\n")
                for dataset, stats in dataset_summary.items():
                    summary_file.write(f"    Dataset: {dataset}\n")
                    summary_file.write(f"      Mean: {stats['mean']}\n")
                    summary_file.write(f"      Std Dev: {stats['std']}\n")
                    summary_file.write(f"      Avg Grouped Std Dev: {stats['grouped_std']}\n")
                
                if model_averages[order_type]['mean'] is not None:
                    summary_file.write(f"  {order_type} Average Across Datasets:\n")
                    summary_file.write(f"    Mean: {model_averages[order_type]['mean']}\n")
                    summary_file.write(f"    Std Dev: {model_averages[order_type]['std']}\n")
                    summary_file.write(f"    Avg Grouped Std Dev: {model_averages[order_type]['grouped_std']}\n\n")
                    
                    # Add to overall calculations
                    overall_means[order_type].append(model_averages[order_type]['mean'])
                    overall_stds[order_type].append(model_averages[order_type]['std'])
                    overall_grouped_stds[order_type].append(model_averages[order_type]['grouped_std'])
            
            summary_file.write("--- END MODEL ---\n\n")

        # Calculate overall statistics by dataset across all models
        summary_file.write("Overall Average Statistics By Dataset Across All Models:\n\n")
        
        # Prepare to collect stats across all datasets first, then iterate by dataset.
        all_dataset_aggregates = defaultdict(lambda: {'example_to_order': {'mean': [], 'std': [], 'grouped_std': []},
                                                      'order_to_example': {'mean': [], 'std': [], 'grouped_std': []}})
        
        for model, results in overall_results.items():
            for order_type in ['example_to_order', 'order_to_example']:
                for dataset, stats in results[order_type].items():
                    all_dataset_aggregates[dataset][order_type]['mean'].append(stats['mean'])
                    all_dataset_aggregates[dataset][order_type]['std'].append(stats['std'])
                    all_dataset_aggregates[dataset][order_type]['grouped_std'].append(stats['grouped_std'])

        for dataset, order_stats in all_dataset_aggregates.items():
            summary_file.write(f"Dataset: {dataset}\n")
            for order_type, stats in order_stats.items():
                if stats['mean']:
                    avg_mean = sum(stats['mean']) / len(stats['mean'])
                    avg_std = sum(stats['std']) / len(stats['std'])
                    avg_grouped_std = sum(stats['grouped_std']) / len(stats['grouped_std'])
                    
                    summary_file.write(f"  {order_type}:\n")
                    summary_file.write(f"    Mean: {avg_mean}\n")
                    summary_file.write(f"    Std Dev: {avg_std}\n")
                    summary_file.write(f"    Avg Grouped Std Dev: {avg_grouped_std}\n\n")
        
        # Calculate overall statistics across all models and datasets
        summary_file.write("Overall Average Statistics Across All Models and Datasets:\n\n")
        for order_type in ['example_to_order', 'order_to_example']:
            if overall_means[order_type]:
                avg_mean = sum(overall_means[order_type]) / len(overall_means[order_type])
                avg_std = sum(overall_stds[order_type]) / len(overall_stds[order_type])
                avg_grouped_std = sum(overall_grouped_stds[order_type]) / len(overall_grouped_stds[order_type])
                
                summary_file.write(f"Average Accuracy (Mean) for {order_type}: {avg_mean}\n")
                summary_file.write(f"Average Standard Deviation for {order_type}: {avg_std}\n")
                summary_file.write(f"Average of Grouped Standard Deviations for {order_type}: {avg_grouped_std}\n\n")


if __name__ == "__main__":
    base_directory = '/data/wyl003/prompt-ordering-research/outputs_new/google'
    pattern_grouping = False
    process_all_csv_files(base_directory, pattern_grouping=pattern_grouping)
