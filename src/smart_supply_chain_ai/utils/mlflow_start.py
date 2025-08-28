import subprocess
import os
import argparse
import platform

def start_mlflow_ui(directory=None):
    '''Starts the MLflow UI server in a separate window/background process.'''
    print("Starting the MLflow UI...")
    
    try:
        if directory:
            print(f"Changing the working directory to: {directory}")
            os.chdir(directory)

        if platform.system() == "Windows":
            print("Detected Windows. Starting UI in a new console window.")
            creation_flags = subprocess.CREATE_NEW_CONSOLE
            command = "pdm run mlflow ui --backend-store-uri file:./mlruns"
            process = subprocess.Popen(command, shell=True, creationflags=creation_flags)
            print("MLflow UI started. Check http://127.0.0.1:5000 in your browser.")
            print("To stop, close the new window.")
        else: # Linux, macOS, and others
            print("Detected Unix-based system. Starting UI in the background.")
            command = "pdm run mlflow ui --backend-store-uri file:./mlruns & disown"
            process = subprocess.Popen(command, shell=True)
            print("MLflow UI started. Check http://127.0.0.1:5000 in your browser.")
            print("The process is running in the background.")

    except FileNotFoundError:
        print("Error: The command 'mlflow' or 'pdm' was not found. Make sure MLflow and pdm are installed and in your PATH.")
    except Exception as e:
        print(f"An error has occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Starts the MLflow UI with an optional working directory.")
    parser.add_argument(
        '-d', '--directory', 
        type=str, 
        help="Path to the working directory where the 'mlruns' folder is located."
    )
    args = parser.parse_args()
    start_mlflow_ui(args.directory)