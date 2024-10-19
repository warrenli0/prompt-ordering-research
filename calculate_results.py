import os
import glob
import pandas as pd

def calculate_grouped_stats(data, group_size=10, pattern_grouping=False):
    """
    Calculates the mean and standard deviation in groupings of `group_size` columns.
    If `pattern_grouping` is True, group columns in the pattern 0, 10, 20, ..., 1, 11, 21, ...
    Returns a list of grouped means and standard deviations, as well as overall mean and std.
    """
    grouped_means = []
    grouped_stds = []
    
    if pattern_grouping:
        # Group columns by the pattern: 0, 10, 20, ..., then 1, 11, 21, ..., etc.
        for i in range(group_size):
            pattern_columns = data.iloc[:, i::group_size]  # Select every `group_size` column starting from `i`
            group_mean = pattern_columns.mean(axis=1)
            group_std = pattern_columns.std(axis=1)
            grouped_means.append(group_mean)
            grouped_stds.append(group_std)
    else:
        # Group columns sequentially in blocks of `group_size`
        for i in range(0, data.shape[1], group_size):
            grouped_data = data.iloc[:, i:i + group_size]
            group_mean = grouped_data.mean(axis=1)
            group_std = grouped_data.std(axis=1)
            grouped_means.append(group_mean)
            grouped_stds.append(group_std)
    
    return grouped_means, grouped_stds

def calculate_stats_for_csv(file_path, pattern_grouping=False):
    """
    Reads a CSV file and calculates the mean and standard deviation
    for each grouping of 10 columns, as well as the overall mean and standard deviation.
    """
    # Read the CSV file
    data = pd.read_csv(file_path, header=None)
    
    # Exclude the first row (header)
    data_without_header = data.iloc[1:]
    
    # Calculate grouped stats
    grouped_means, grouped_stds = calculate_grouped_stats(data_without_header, pattern_grouping=pattern_grouping)
    
    # Calculate overall mean and standard deviation for the entire file
    overall_mean = data_without_header.mean(axis=1)
    overall_std = data_without_header.std(axis=1)
    
    # Calculate the average of the standard deviations across all groupings
    avg_grouped_std = pd.concat(grouped_stds, axis=1).mean(axis=1).mean()
    
    return data_without_header, grouped_means, grouped_stds, overall_mean, overall_std, avg_grouped_std

def process_all_csv_files(base_directory, pattern_grouping=False):
    """
    Traverse the given base directory and process each CSV file found.
    """
    csv_files = glob.glob(os.path.join(base_directory, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print("No CSV files found.")
        return
    
    # List to store all data for calculating overall statistics across all files
    all_data = []
    
    summary_file_path = os.path.join(base_directory, 'summary_results.txt')
    
    with open(summary_file_path, 'w') as summary_file:
        # Collect overall results for all files
        overall_means_list = []
        overall_stds_list = []
        avg_grouped_stds_list = []

        for file_path in csv_files:
            print(f"Processing file: {file_path}")
            data, grouped_means, grouped_stds, overall_mean, overall_std, avg_grouped_std = calculate_stats_for_csv(
                file_path, pattern_grouping=pattern_grouping)
            
            # Append data to all_data list for overall statistics
            all_data.append(data)
            overall_means_list.append(overall_mean.mean())
            overall_stds_list.append(overall_std.mean())
            avg_grouped_stds_list.append(avg_grouped_std)
            
            # Write results for the current file to the summary file
            summary_file.write(f"File: {file_path}\n")
            
            # Write grouped means and standard deviations
            for i, (group_mean, group_std) in enumerate(zip(grouped_means, grouped_stds)):
                if pattern_grouping:
                    summary_file.write(f"Group {i+1} (columns {i}, {i+10}, {i+2*10}, ...):\n")
                else:
                    # Correcting the way columns are displayed for sequential groups
                    start_col = i * 10
                    end_col = start_col + 9 if start_col + 9 < data.shape[1] else data.shape[1] - 1
                    summary_file.write(f"Group {i+1} (columns {start_col+1} to {end_col+1}):\n")
                
                summary_file.write(f"Mean:\n{group_mean.mean()}\n")
                summary_file.write(f"Standard Deviation:\n{group_std.mean()}\n\n")
            
            # Write overall statistics for the current file
            summary_file.write("Overall Mean for the file:\n")
            summary_file.write(f"{overall_mean.mean()}\n")
            summary_file.write("Overall Standard Deviation for the file:\n")
            summary_file.write(f"{overall_std.mean()}\n")
            summary_file.write(f"Average of Grouped Standard Deviations:\n{avg_grouped_std}\n")
            summary_file.write("--- END FILE ---\n\n")
        
        # Concatenate all data from all files into a single DataFrame
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Calculate overall mean and std across all files
        overall_mean_all_files = combined_data.mean(axis=1)
        overall_std_all_files = combined_data.std(axis=1)
        avg_grouped_std_all_files = sum(avg_grouped_stds_list) / len(avg_grouped_stds_list)
        
        # Write overall statistics across all files to the summary file
        summary_file.write("Overall Statistics Across All Files (excluding headers):\n")
        summary_file.write(f"Overall Mean for all rows: \n{overall_mean_all_files.mean()}\n")
        summary_file.write(f"Overall Standard Deviation for all rows: \n{overall_std_all_files.mean()}\n")
        summary_file.write(f"Average of Grouped Standard Deviations Across All Files: \n{avg_grouped_std_all_files}\n")

if __name__ == "__main__":
    base_directory = '/data/wyl003/prompt-ordering-research/outputs_new/results/order_to_example_fixed'
    pattern_grouping = False  # Set to True to use pattern grouping (0, 10, 20, ...), or False for sequential
    process_all_csv_files(base_directory, pattern_grouping=pattern_grouping)
