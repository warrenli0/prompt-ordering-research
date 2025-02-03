import os
import pandas as pd

best_ordering_path = '/data/wyl003/prompt-ordering-research/outputs/best_ordering/'
train_ordering_path = '/data/wyl003/prompt-ordering-research/outputs/train_ordering/'

# Data structures to hold aggregated data
model_data = {}
dataset_data = {}

# Loop through subfolders in best_ordering_path (e.g., 'google')
for subfolder in os.listdir(best_ordering_path):
    if subfolder == 'Qwen': continue
    print(subfolder)
    subfolder_path_best = os.path.join(best_ordering_path, subfolder)
    subfolder_path_train = os.path.join(train_ordering_path, subfolder)
    
    if os.path.isdir(subfolder_path_best):
        # Now, loop through models within each subfolder
        for model_name in os.listdir(subfolder_path_best):
            model_path_best = os.path.join(subfolder_path_best, model_name)
            model_path_train = os.path.join(subfolder_path_train, model_name)
            
            # Proceed only if the corresponding train_ordering path exists
            if os.path.exists(model_path_train):
                # Use a unique key for the model (including subfolder)
                model_key = os.path.join(subfolder, model_name)
                model_data[model_key] = {
                    'Best Ordering Train Accuracy': [],
                    'Final Evaluation Accuracy': [],
                    'Majority Vote Accuracy': [],
                    'Best Ordering Accuracy': []
                }
                
                # Paths to the 'results' directories
                results_path_best = os.path.join(model_path_best, 'results')
                results_path_train = os.path.join(model_path_train, 'results')
                
                if os.path.exists(results_path_best) and os.path.exists(results_path_train):
                    # Process dataset files within the 'results' directory
                    for dataset_file in os.listdir(results_path_best):
                        # print(dataset_file)
                        if dataset_file.endswith('_accuracy_results.csv'):
                            dataset_name = dataset_file.replace('_accuracy_results.csv', '')
                            
                            # Paths to the CSV files
                            best_ordering_csv = os.path.join(results_path_best, dataset_file)
                            train_ordering_csv = os.path.join(
                                results_path_train,
                                f"{dataset_name}_final_best_orderings_evaluation.csv"
                            )
                            
                            if os.path.exists(train_ordering_csv):
                                # Read CSV files
                                df_best = pd.read_csv(best_ordering_csv)
                                df_train = pd.read_csv(train_ordering_csv)
                                
                                # Aggregate data for the model
                                model_data[model_key]['Majority Vote Accuracy'].extend(
                                    df_best['Majority Vote Accuracy']
                                )
                                model_data[model_key]['Best Ordering Accuracy'].extend(
                                    df_best['Best Ordering Accuracy']
                                )
                                model_data[model_key]['Best Ordering Train Accuracy'].extend(
                                    df_train['Best Ordering Train Accuracy']
                                )
                                model_data[model_key]['Final Evaluation Accuracy'].extend(
                                    df_train['Final Evaluation Accuracy']
                                )
                                
                                # Initialize dataset data if not already
                                if dataset_name not in dataset_data:
                                    dataset_data[dataset_name] = {
                                        'Best Ordering Train Accuracy': [],
                                        'Final Evaluation Accuracy': [],
                                        'Majority Vote Accuracy': [],
                                        'Best Ordering Accuracy': []
                                    }
                                
                                # Aggregate data for the dataset
                                dataset_data[dataset_name]['Majority Vote Accuracy'].extend(
                                    df_best['Majority Vote Accuracy']
                                )
                                dataset_data[dataset_name]['Best Ordering Accuracy'].extend(
                                    df_best['Best Ordering Accuracy']
                                )
                                dataset_data[dataset_name]['Best Ordering Train Accuracy'].extend(
                                    df_train['Best Ordering Train Accuracy']
                                )
                                dataset_data[dataset_name]['Final Evaluation Accuracy'].extend(
                                    df_train['Final Evaluation Accuracy']
                                )

# Compute averages for models
model_averages = []
for model_key, metrics in model_data.items():
    avg_data = {
        'Model': model_key,
        'Average Best Ordering Train Accuracy': sum(metrics['Best Ordering Train Accuracy']) / len(metrics['Best Ordering Train Accuracy']),
        'Average Final Evaluation Accuracy': sum(metrics['Final Evaluation Accuracy']) / len(metrics['Final Evaluation Accuracy']),
        'Average Majority Vote Accuracy': sum(metrics['Majority Vote Accuracy']) / len(metrics['Majority Vote Accuracy']),
        'Average Best Ordering Accuracy': sum(metrics['Best Ordering Accuracy']) / len(metrics['Best Ordering Accuracy'])
    }
    model_averages.append(avg_data)

# Compute averages for datasets
dataset_averages = []
for dataset_name, metrics in dataset_data.items():
    avg_data = {
        'Dataset': dataset_name,
        'Average Best Ordering Train Accuracy': sum(metrics['Best Ordering Train Accuracy']) / len(metrics['Best Ordering Train Accuracy']),
        'Average Final Evaluation Accuracy': sum(metrics['Final Evaluation Accuracy']) / len(metrics['Final Evaluation Accuracy']),
        'Average Majority Vote Accuracy': sum(metrics['Majority Vote Accuracy']) / len(metrics['Majority Vote Accuracy']),
        'Average Best Ordering Accuracy': sum(metrics['Best Ordering Accuracy']) / len(metrics['Best Ordering Accuracy'])
    }
    dataset_averages.append(avg_data)

# DataFrame for models
df_models = pd.DataFrame(model_averages)
df_models = df_models[['Model', 'Average Best Ordering Train Accuracy', 'Average Final Evaluation Accuracy', 
                       'Average Majority Vote Accuracy', 'Average Best Ordering Accuracy']]

# DataFrame for datasets
df_datasets = pd.DataFrame(dataset_averages)
df_datasets = df_datasets[['Dataset', 'Average Best Ordering Train Accuracy', 'Average Final Evaluation Accuracy', 
                           'Average Majority Vote Accuracy', 'Average Best Ordering Accuracy']]

# Export models data
df_models.to_excel('google_majority_vote_align_models_summary.xlsx', index=False)

# Export datasets data
df_datasets.to_excel('google_majority_vote_align_datasets_summary.xlsx', index=False)
