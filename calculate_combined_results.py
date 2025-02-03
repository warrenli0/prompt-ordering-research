import os
import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_stats(data, group_size):
    """Calculate statistics for a given data array and group size"""
    if len(data) == 0:
        return None
    
    # Calculate overall statistics
    overall_avg = np.mean(data)
    overall_std = np.std(data, ddof=1)
    
    # Calculate set-wise statistics
    set_averages = []
    for i in range(0, len(data), group_size):
        group = data[i:i+group_size]
        set_averages.append(np.mean(group))
    
    set_avg_std = np.std(set_averages, ddof=1) if set_averages else 0
    
    return {
        'average': overall_avg,
        'std_dev': overall_std,
        'set_avg_std_dev': set_avg_std
    }

def process_experiment_results(base_path):
    # Define experiment parameters
    experiments = ['example_to_order', 'majority_vote', 'best_ordering', 'train_ordering']
    group_sizes = {
        'example_to_order': 10,
        'majority_vote': 1,
        'best_ordering': 1,
        'train_ordering': 1
    }
    
    # Initialize data structures
    model_data = defaultdict(lambda: defaultdict(list))
    dataset_data = defaultdict(lambda: defaultdict(list))
    model_dataset_data = []

    # Walk through directory structure
    for root, dirs, files in os.walk(base_path):
        if 'results' in root:
            for file in files:
                if file.endswith('_results.csv'):
                    # Parse path information
                    path_parts = root.split(os.sep)
                    model = path_parts[3]
                    experiment = path_parts[5]
                    dataset = file.replace('_results.csv', '')
                    
                    # Read CSV file
                    df = pd.read_csv(os.path.join(root, file))
                    accuracies = df['Accuracy'].tolist()
                    
                    # Calculate statistics
                    stats = calculate_stats(accuracies, group_sizes[experiment])
                    
                    if stats:
                        # Store model data
                        model_data[model][experiment].append(stats)
                        
                        # Store dataset data
                        dataset_data[dataset][experiment].append(stats)
                        
                        # Store model-dataset combination data
                        model_dataset_data.append({
                            'Model': model,
                            'Dataset': dataset,
                            'Experiment': experiment,
                            **stats
                        })

    return model_data, dataset_data, model_dataset_data

def create_summary_dataframes(model_data, dataset_data, model_dataset_data):
    # Process model data
    model_rows = []
    for model, experiments in model_data.items():
        row = {'Model': model}
        for exp, stats_list in experiments.items():
            if stats_list:
                avg = np.mean([s['average'] for s in stats_list])
                std = np.mean([s['std_dev'] for s in stats_list])
                set_std = np.mean([s['set_avg_std_dev'] for s in stats_list])
                
                row[f'{exp}_avg'] = avg
                row[f'{exp}_std'] = std
                row[f'{exp}_set_std'] = set_std
        model_rows.append(row)

    # Process dataset data
    dataset_rows = []
    for dataset, experiments in dataset_data.items():
        row = {'Dataset': dataset}
        for exp, stats_list in experiments.items():
            if stats_list:
                avg = np.mean([s['average'] for s in stats_list])
                std = np.mean([s['std_dev'] for s in stats_list])
                set_std = np.mean([s['set_avg_std_dev'] for s in stats_list])
                
                row[f'{exp}_avg'] = avg
                row[f'{exp}_std'] = std
                row[f'{exp}_set_std'] = set_std
        dataset_rows.append(row)

    # Process model-dataset data
    model_dataset_df = pd.DataFrame(model_dataset_data)
    
    return pd.DataFrame(model_rows), pd.DataFrame(dataset_rows), model_dataset_df

def save_to_excel(model_df, dataset_df, model_dataset_df):
    # Create output directory
    os.makedirs('analysis_results', exist_ok=True)
    
    # Save model-based results
    model_df.to_excel('analysis_results/model_summary_32.xlsx', index=False)
    
    # Save dataset-based results
    dataset_df.to_excel('analysis_results/dataset_summary_32.xlsx', index=False)
    
    # Save model-dataset combined results
    model_dataset_df.to_excel('analysis_results/model_dataset_summary_32.xlsx', index=False)

def main():
    base_path = 'outputs/combined'
    model_data, dataset_data, model_dataset_data = process_experiment_results(base_path)
    model_df, dataset_df, model_dataset_df = create_summary_dataframes(model_data, dataset_data, model_dataset_data)
    save_to_excel(model_df, dataset_df, model_dataset_df)

if __name__ == '__main__':
    main()