import os
import glob
import pandas as pd

def reformat_csv(file_path, output_path):
    """
    Reformats a 10x10 CSV file into a 1x100 CSV by flattening the rows into a single row.
    Removes the original header (0-9) and adds a new header that ranges from 0 to 99.
    """
    # Read the CSV file, skipping the first row (original header)
    data = pd.read_csv(file_path, header=0)  # This will skip the original 0-9 header
    
    # Flatten the data: convert the DataFrame into a single row
    flattened_data = data.to_numpy().flatten()
    
    # Convert the flattened array back to a DataFrame with 1 row and 100 columns
    reformatted_data = pd.DataFrame([flattened_data])
    
    # Create the header as a list of column numbers from 0 to 99
    header = list(range(100))
    
    # Save the reformatted data to a new CSV file with the correct header
    reformatted_data.to_csv(output_path, index=False, header=header)

def reformat_all_csv_files(base_directory, output_directory):
    """
    Traverse the given base directory, reformat each CSV file, and save it to the output directory.
    """
    csv_files = glob.glob(os.path.join(base_directory, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print("No CSV files found.")
        return
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for file_path in csv_files:
        print(f"Reformatting file: {file_path}")
        # Generate output file path
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_directory, file_name)
        
        # Reformat the CSV file
        reformat_csv(file_path, output_path)
        print(f"Saved reformatted file to: {output_path}")

if __name__ == "__main__":
    base_directory = '/data/wyl003/prompt-ordering-research/outputs_new/results/order_to_example'
    output_directory = '/data/wyl003/prompt-ordering-research/outputs_new/results/order_to_example_fixed'
    
    # Reformat all CSV files in the base directory
    reformat_all_csv_files(base_directory, output_directory)
