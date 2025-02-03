import os
import glob
import pandas as pd
from collections import defaultdict

def calculate_stats_for_csv(file_path):
    data = pd.read_csv(file_path, header=None)
    data_without_header = data.iloc[1:]  # Assuming the first row is a header

    overall_mean = data_without_header.mean(axis=1).mean()
    overall_std = data_without_header.std(axis=1).mean()

    return overall_mean, overall_std

def process_csv_files_in_subfolder(subfolder):
    dataset_results = defaultdict(lambda: {'mean': [], 'std': []})
    csv_files = glob.glob(os.path.join(subfolder, '*.csv'))
    
    for file_path in csv_files:
        dataset_name = os.path.basename(file_path).split('.')[0]
        overall_mean, overall_std = calculate_stats_for_csv(file_path)
        
        dataset_results[dataset_name]['mean'].append(overall_mean)
        dataset_results[dataset_name]['std'].append(overall_std)

    return dataset_results

def summarize_dataset_results(dataset_results):
    summary = {}
    for dataset, stats in dataset_results.items():
        if stats['mean']:
            summary[dataset] = {
                'mean': sum(stats['mean']) / len(stats['mean']),
                'std': sum(stats['std']) / len(stats['std']),
            }
    return summary

def calculate_averages_across_datasets(dataset_summary):
    all_means = [stats['mean'] for stats in dataset_summary.values()]
    all_stds = [stats['std'] for stats in dataset_summary.values()]

    if all_means:
        return {
            'mean': sum(all_means) / len(all_means),
            'std': sum(all_stds) / len(all_stds),
        }
    else:
        return {
            'mean': None,
            'std': None,
        }

def summarize_dataset_over_models(datasets_overall_stats):
    dataset_overall_summary = {}
    for dataset, stats in datasets_overall_stats.items():
        if stats['mean']:
            dataset_overall_summary[dataset] = {
                'mean': sum(stats['mean']) / len(stats['mean']),
                'std': sum(stats['std']) / len(stats['std']),
            }
    return dataset_overall_summary

def process_all_csv_files(base_directory):
    subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) 
                      if os.path.isdir(os.path.join(base_directory, d))]
    
    summary_file_path = os.path.join(base_directory, 'summary_results.txt')
    overall_means = []
    overall_stds = []
    datasets_overall_stats = defaultdict(lambda: {'mean': [], 'std': []})

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
            
            summary_file.write(f"  Results:\n")
            for dataset, stats in dataset_summary.items():
                summary_file.write(f"    Dataset: {dataset}\n")
                summary_file.write(f"      Mean: {stats['mean']}\n")
                summary_file.write(f"      Std Dev: {stats['std']}\n")

                # Collect dataset stats across models
                datasets_overall_stats[dataset]['mean'].append(stats['mean'])
                datasets_overall_stats[dataset]['std'].append(stats['std'])
                
            # Calculate model-level averages across datasets
            model_averages = calculate_averages_across_datasets(dataset_summary)
            if model_averages['mean'] is not None:
                summary_file.write(f"  Average Across Datasets:\n")
                summary_file.write(f"    Mean: {model_averages['mean']}\n")
                summary_file.write(f"    Std Dev: {model_averages['std']}\n\n")
                
                overall_means.append(model_averages['mean'])
                overall_stds.append(model_averages['std'])
            
            summary_file.write("--- END MODEL ---\n\n")

        # After processing all models, output the dataset grouped results
        summary_file.write("Summary of results grouped by dataset:\n\n")
        dataset_overall_summary = summarize_dataset_over_models(datasets_overall_stats)
        for dataset, stats in dataset_overall_summary.items():
            summary_file.write(f"Dataset: {dataset}\n")
            summary_file.write(f"  Mean Across Models: {stats['mean']}\n")
            summary_file.write(f"  Std Dev Across Models: {stats['std']}\n\n")
        
        summary_file.write("Overall Average Statistics Across All Models:\n\n")
        if overall_means:
            avg_mean = sum(overall_means) / len(overall_means)
            avg_std = sum(overall_stds) / len(overall_stds)
            
            summary_file.write(f"Average Accuracy (Mean): {avg_mean}\n")
            summary_file.write(f"Average Standard Deviation: {avg_std}\n\n")

if __name__ == "__main__":
    base_directory = '/data/wyl003/prompt-ordering-research/outputs/majority_vote/google'
    process_all_csv_files(base_directory)
