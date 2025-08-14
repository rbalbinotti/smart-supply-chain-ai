# File: src/smart_supply_chain_ai/data_processing/get_data.py

# Import necessary libraries
import os
import shutil
import zipfile
import glob
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate the API
# Kaggle token is required
api = KaggleApi()
api.authenticate()

# Define the main function to download the dataset
def download_kaggle_dataset(dataset_name: str, destination_path: str):
    """
    Downloads a dataset from Kaggle, saves it to a local folder, and removes the zip file.

    Args:
        dataset_name (str): The name of the dataset on Kaggle (e.g., 'grocery/grocery-sales-data').
        destination_path (str): The path to the folder where the dataset will be saved.
    """
    print(f"Starting the download of dataset '{dataset_name}' from Kaggle...")
    try:
        # Download the zip file from Kaggle, but do not unzip it (unzip=False is the default)
        api.dataset_download_files(dataset_name, path=destination_path, force=False)

        # Find the downloaded zip file in the destination folder
        search_pattern = os.path.join(destination_path, '*.zip')
        zip_files = glob.glob(search_pattern)

        if not zip_files:
            print("No .zip file was found after download. The download may have failed.")
            return

        # Get the first (and only) zip file path from the list
        zip_file_path = zip_files[0]
        
        # Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_path)
        
        # Remove the original zip file
        os.remove(zip_file_path)
                
        print(f"Download, unzipping, and cleanup complete! The dataset was saved to: {destination_path}")
        
    except Exception as e:
        print(f"An error occurred during the dataset download: {e}")

# Example of how to use the function
if __name__ == "__main__":
    DATASET_NAME = "salahuddinahmedshuvo/grocery-inventory-and-sales-dataset"
    
    # Calculate the absolute path to the project root
    # __file__ -> '.../src/smart_supply_chain_ai/data_processing/get_data.py'
    # dirname(__file__) -> '.../src/smart_supply_chain_ai/data_processing'
    # '..' -> '.../src/smart_supply_chain_ai'
    # '..' -> '.../src'
    # '..' -> '...' (project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # Construct the final destination path
    DESTINATION_DIR = os.path.join(project_root, 'data', 'raw')
    
    # Ensure the destination directory exists
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    
    download_kaggle_dataset(DATASET_NAME, DESTINATION_DIR)