import os
import pandas as pd

entropy_path = '/data/wyl003/prompt-ordering-research/outputs/entropy_baseline/'

# Data structure to store results grouped by model and dataset
results = {}

# Loop through subfolders in entropy_path (e.g., 'google')
for subfolder in os.listdir(entropy_path):
    subfolder_path = os.path.join(entropy_path, subfolder)
    
    if os.path.isdir(subfolder_path):
        # Loop through models within each subfolder
        for model_name in os.listdir(subfolder_path):
            model_path = os.path.join(subfolder_path, model_name)
            
            if os.path.isdir(model_path):
                # Use a unique key for the model (including subfolder)
                model_key = os.path.join(subfolder, model_name)
                
                # Initialize results for this model
                if model_key not in results:
                    results[model_key] = {}

                # Path to the 'results' directory
                results_path = os.path.join(model_path, 'results')
                
                if os.path.exists(results_path):
                    # Process dataset files within the 'results' directory
                    for dataset_file in os.listdir(results_path):
                        if dataset_file.endswith('.csv'):
                            dataset_name = dataset_file.replace('.csv', '')
                            dataset_csv = os.path.join(results_path, dataset_file)
                            
                            # Read CSV file without header
                            df = pd.read_csv(dataset_csv, header=None)
                            accuracies = df.iloc[1]  # The only row contains the accuracies

                            # Calculate average and standard deviation
                            avg_accuracy = accuracies.mean()
                            std_accuracy = accuracies.std()

                            # Store the results
                            results[model_key][dataset_name] = {
                                'Average Accuracy': avg_accuracy,
                                'STD Accuracy': std_accuracy
                            }

# Flatten results for each model-dataset pair
flattened_results = []
for model, datasets in results.items():
    for dataset, metrics in datasets.items():
        flattened_results.append({
            'Model': model,
            'Dataset': dataset,
            'Average Accuracy': metrics['Average Accuracy'],
            'STD Accuracy': metrics['STD Accuracy']
        })

df_results = pd.DataFrame(flattened_results)

# Save detailed results
df_results.to_excel('entropy_model_dataset_summary.xlsx', index=False)

# Calculate grouped averages and standard deviations by model
model_grouped = df_results.groupby('Model').agg(
    {
        'Average Accuracy': ['mean', 'std'],
        'STD Accuracy': ['mean', 'std']
    }
).reset_index()
model_grouped.columns = ['Model', 'Avg Accuracy (Mean)', 'Avg Accuracy (STD)', 'STD Accuracy (Mean)', 'STD Accuracy (STD)']

# Calculate grouped averages and standard deviations by dataset
dataset_grouped = df_results.groupby('Dataset').agg(
    {
        'Average Accuracy': ['mean', 'std'],
        'STD Accuracy': ['mean', 'std']
    }
).reset_index()
dataset_grouped.columns = ['Dataset', 'Avg Accuracy (Mean)', 'Avg Accuracy (STD)', 'STD Accuracy (Mean)', 'STD Accuracy (STD)']

# Save grouped averages
with pd.ExcelWriter('entropy_grouped_summary.xlsx') as writer:
    model_grouped.to_excel(writer, sheet_name='By Model', index=False)
    dataset_grouped.to_excel(writer, sheet_name='By Dataset', index=False)

# Print summary for verification
print("Detailed results saved to 'entropy_model_dataset_summary.xlsx'")
print("Grouped summaries saved to 'entropy_grouped_summary.xlsx'")
