#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import sys
import numpy as np
import tensorflow as tf
import subprocess
import time
from colorama import Fore, Style, init

# Initialize colorama for colored console output
init(autoreset=True)

def print_header():
    """Print header for the SST bias correction tool."""
    header = f"""
{Fore.CYAN}============================================================
{Fore.CYAN}||       SEA SURFACE TEMPERATURE BIAS CORRECTION          ||
{Fore.CYAN}============================================================{Style.RESET_ALL}
    """
    print(header)

def print_section(title):
    """Print a section title."""
    print(f"\n{Fore.YELLOW}=== {title} ==={Style.RESET_ALL}\n")

def print_success(message):
    """Print a success message."""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_info(message):
    """Print an information message."""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")

def print_error(message):
    """Print an error message."""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def check_gpu_availability():
    """Check if GPU is available for TensorFlow."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print_success(f"Found {len(gpus)} GPU(s): {', '.join([gpu.name for gpu in gpus])}")
        # Get GPU details
        for gpu in gpus:
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print_info(f"  - {gpu.name}: {gpu_details.get('device_name', 'Unknown')}")
            except:
                pass
        return True
    else:
        print_info("No GPUs found. Training will run on CPU.")
        return False

def check_data_availability(method):
    """Check if data files for the SST variable exist based on the correction method."""
    
    # Define required base directories based on the method
    if method == "unet":
        base_dir = "../data/sst/"
        essential_files = [
            f"{base_dir}cmip6_sst_1958_2014_fill_diststen.mat",
            f"{base_dir}oras5_sst_1958_2014_fill_diststen.mat",
            f"{base_dir}oras5_historical_sst_1958_2020_mean.mat"
        ]
    elif method == "bilstm":
        base_dir = "../data/thetao/"
        essential_files = [
            f"{base_dir}cmip6_thetao_1958_2014_2d.npy",
            f"{base_dir}cmip6_thetao_transpose_ts_1958_2014_2d.npy",
            f"{base_dir}oras5_temp_1958_2014_2d.npy"
        ]
    elif method == "convlstm":
        base_dir = "../data/thetao/"
        essential_files = [
            f"{base_dir}cmip6_thetao_1958_2014_fill_diststen.mat",
            f"{base_dir}oras5_sst_1958_2014_fill_diststen.mat"
        ]
    
    # Create directories if they don't exist
    required_dirs = [base_dir, "../output/", "../output/models/"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print_info(f"Created directory: {directory}")
    
    # Check for missing files
    missing_files = [f for f in essential_files if not os.path.exists(f)]
    
    if missing_files:
        print_error("Missing required data files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    else:
        print_success(f"All required data files for SST using {method} are available")
        return True

def list_correction_methods():
    """List available correction methods for SST."""
    print_section("Available Correction Methods for SST")
    
    methods = {
        "unet": "U-Net CNN architecture for spatial bias correction",
        "bilstm": "Bidirectional LSTM for temporal bias correction",
        "convlstm": "Convolutional LSTM for spatiotemporal bias correction"
    }
    
    for idx, (method, description) in enumerate(methods.items(), 1):
        print(f"{idx}. {Fore.CYAN}{method}{Style.RESET_ALL}: {description}")
    
    return methods

def run_correction(method, epochs=None, batch_size=None, use_existing_model=False):
    """Run the selected correction method for SST."""
    print_section(f"Running {method.upper()} correction for Sea Surface Temperature")
    
    # Determine the correct script to run based on the method
    if method == "unet":
        script_path = "./sst_unet_reorganised.py"
    elif method == "bilstm":
        script_path = "./sst_bilstm.py"
    elif method == "convlstm":
        script_path = "./sst_convlstm.py"
    else:
        print_error(f"Unknown method: {method}")
        return False
    
    # Check if script file exists
    if not os.path.exists(script_path):
        print_error(f"Script file not found: {script_path}")
        return False
    
    # Modify epochs, batch_size, and use_existing_model in the script if provided
    if epochs is not None or batch_size is not None or use_existing_model:
        try:
            with open(script_path, 'r') as file:
                script_contents = file.read()
            
            # Replace epochs value if provided
            if epochs is not None:
                script_contents = script_contents.replace("epochs=500", f"epochs={epochs}")
                script_contents = script_contents.replace("epochs=1000", f"epochs={epochs}")
                script_contents = script_contents.replace("epochs=2000", f"epochs={epochs}")
                script_contents = script_contents.replace("epochs=2500", f"epochs={epochs}")
            
            # Replace batch_size value if provided
            if batch_size is not None:
                script_contents = script_contents.replace("batch_size=32", f"batch_size={batch_size}")
                script_contents = script_contents.replace("batch_size=64", f"batch_size={batch_size}")
            
            # Modify to use existing model if required
            if use_existing_model:
                if method == "bilstm":
                    script_contents = script_contents.replace("use_existing_model = False", "use_existing_model = True")
                elif method == "convlstm":
                    script_contents = script_contents.replace("model = create_convlstm_model()", 
                                                             "# Load existing model\ntry:\n    model = keras.models.load_model('../output/models/convlstm_model.keras')\n    print('Loaded existing model')\nexcept:\n    print('No existing model found, creating new model')\n    model = create_convlstm_model()")
                elif method == "unet":
                    # For UNet, add model loading before the create_model call
                    model_creation_line = "model = create_unet_model()"
                    model_loading_code = """try:
    print('Attempting to load existing UNet model...')
    model = keras.models.load_model('../output/models/unet_sst_model.h5', custom_objects={"mse_loss": custom_mse_loss(mask)})
    print('Loaded existing model')
except Exception as e:
    print('Creating new UNet model:', e)
    model = create_unet_model()"""
                    script_contents = script_contents.replace(model_creation_line, model_loading_code)
            
            # Write modified content to a temporary file
            temp_script_path = f"./temp_{method}_sst.py"
            with open(temp_script_path, 'w') as file:
                file.write(script_contents)
            
            script_path = temp_script_path
            
        except Exception as e:
            print_error(f"Error modifying script parameters: {str(e)}")
            return False
    
    # Run the script
    try:
        start_time = time.time()
        print_info(f"Starting execution of {script_path}")
        process = subprocess.Popen([sys.executable, script_path], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   universal_newlines=True)
        
        # Print output in real-time
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line.strip())
        process.stdout.close()
        
        # Handle errors
        for stderr_line in iter(process.stderr.readline, ""):
            print_error(stderr_line.strip())
        process.stderr.close()
        
        return_code = process.wait()
        end_time = time.time()
        
        # Clean up temporary file if created
        if epochs is not None or batch_size is not None or use_existing_model:
            try:
                os.remove(temp_script_path)
            except:
                pass
        
        if return_code == 0:
            elapsed_time = end_time - start_time
            print_success(f"SST correction completed successfully in {elapsed_time:.2f} seconds")
            return True
        else:
            print_error(f"SST correction failed with return code {return_code}")
            return False
        
    except Exception as e:
        print_error(f"Error executing script: {str(e)}")
        return False

def conduct_ablation_study(method, epochs=100, batch_sizes=[32, 64, 128]):
    """Conduct an ablation study with different hyperparameters."""
    print_section(f"Conducting Ablation Study for SST using {method.upper()}")
    
    results = {}
    results[method] = {}
    
    for batch_size in batch_sizes:
        print_info(f"Testing batch size: {batch_size}")
        success = run_correction(method, epochs=epochs, batch_size=batch_size)
        results[method][batch_size] = "Success" if success else "Failed"
    
    # Print summary of ablation study
    print_section("Ablation Study Results")
    for batch_size, result in results[method].items():
        color = Fore.GREEN if result == "Success" else Fore.RED
        print(f"Batch size {batch_size}: {color}{result}{Style.RESET_ALL}")
    
    return results

def main():
    """Main function to run the SST correction toolkit."""
    parser = argparse.ArgumentParser(description="Sea Surface Temperature Bias Correction Toolkit")
    parser.add_argument("--method", type=str, help="Correction method (unet, bilstm, convlstm)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--load_model", action="store_true", help="Load an existing model instead of training")
    parser.add_argument("--ablation", action="store_true", help="Conduct ablation study")
    
    args = parser.parse_args()
    
    print_header()
    check_gpu_availability()
    
    methods = list_correction_methods()
    
    # If arguments were provided, use them
    if args.method:
        if args.method not in methods:
            print_error(f"Unknown method: {args.method}")
            return
        
        if args.ablation:
            check_data_availability(args.method)
            conduct_ablation_study(args.method, 
                                  epochs=args.epochs or 100, 
                                  batch_sizes=[args.batch_size] if args.batch_size else [32, 64, 128])
        else:
            check_data_availability(args.method)
            run_correction(args.method, args.epochs, args.batch_size, use_existing_model=args.load_model)
    
    # Otherwise, use interactive mode
    else:
        # Get method choice
        method_choice = None
        while method_choice is None:
            try:
                choice = int(input(f"\n{Fore.BLUE}Select correction method (1-{len(methods)}): {Style.RESET_ALL}"))
                if 1 <= choice <= len(methods):
                    method_choice = list(methods.keys())[choice-1]
                else:
                    print_error("Invalid choice. Please try again.")
            except ValueError:
                print_error("Please enter a number.")
        
        # Check data availability with the selected method
        check_data_availability(method_choice)
        
        # Ask about using existing model
        use_existing_model = input(f"\n{Fore.BLUE}Use existing model? (y/n): {Style.RESET_ALL}").lower() == 'y'
        
        # Ask about ablation study
        ablation_choice = input(f"\n{Fore.BLUE}Conduct ablation study? (y/n): {Style.RESET_ALL}").lower() == 'y'
        
        if ablation_choice:
            # Get epochs for ablation study
            epochs = None
            while epochs is None:
                try:
                    epochs = int(input(f"\n{Fore.BLUE}Enter number of epochs for the study: {Style.RESET_ALL}"))
                    if epochs <= 0:
                        print_error("Epochs must be positive.")
                        epochs = None
                except ValueError:
                    print_error("Please enter a number.")
            
            # Conduct ablation study
            conduct_ablation_study(method_choice, epochs=epochs)
        else:
            # Get epochs for single run
            epochs = None
            epochs_input = input(f"\n{Fore.BLUE}Enter number of epochs (press Enter for default): {Style.RESET_ALL}")
            if epochs_input:
                try:
                    epochs = int(epochs_input)
                    if epochs <= 0:
                        print_error("Epochs must be positive. Using default.")
                        epochs = None
                except ValueError:
                    print_error("Invalid value. Using default.")
            
            # Get batch size for single run
            batch_size = None
            batch_size_input = input(f"\n{Fore.BLUE}Enter batch size (press Enter for default): {Style.RESET_ALL}")
            if batch_size_input:
                try:
                    batch_size = int(batch_size_input)
                    if batch_size <= 0:
                        print_error("Batch size must be positive. Using default.")
                        batch_size = None
                except ValueError:
                    print_error("Invalid value. Using default.")
            
            # Run correction
            run_correction(method_choice, epochs, batch_size, use_existing_model=use_existing_model)

if __name__ == "__main__":
    main()
