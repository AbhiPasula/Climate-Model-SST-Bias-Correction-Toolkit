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
    """Print header for the climate model bias correction tool."""
    header = f"""
{Fore.CYAN}============================================================
{Fore.CYAN}||       GLOBAL CLIMATE MODEL ERROR CORRECTION TOOLKIT     ||
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

def check_data_availability(variable):
    """Check if data files for the specified variable exist."""
    
    # Define required base directories based on the variable
    if variable == "sst":
        base_dir = "../data/sst/"
        essential_files = [
            f"{base_dir}cmip6_sst_1958_2014_fill_diststen.mat",
            f"{base_dir}oras5_sst_1958_2014_fill_diststen.mat",
            f"{base_dir}oras5_historical_sst_1958_2020_mean.mat"
        ]
    elif variable == "dsl":
        base_dir = "../data/zos/"  # DSL is stored as zos in data structure
        essential_files = [
            f"{base_dir}cmip6_zos_1958_2014_fill_diststen.mat",
            f"{base_dir}oras5_zos_1958_2014_fill_diststen.mat",
            f"{base_dir}oras5_historical_zos_1958_2020_mean.mat"
        ]
    else:
        print_error(f"Unknown variable: {variable}")
        return False
    
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
        print_success(f"All required data files for {variable.upper()} are available")
        return True

def list_variables():
    """List available variables for bias correction."""
    print_section("Available Variables for Bias Correction")
    
    variables = {
        "sst": "Sea Surface Temperature (SST)",
        "dsl": "Dynamic Sea Level (DSL)"
    }
    
    for idx, (var_code, var_name) in enumerate(variables.items(), 1):
        print(f"{idx}. {Fore.CYAN}{var_code}{Style.RESET_ALL}: {var_name}")
    
    return variables

def run_correction(variable, epochs=None, batch_size=None, use_existing_model=False):
    """Run the UNet correction method for the specified variable."""
    print_section(f"Running UNet correction for {variable.upper()}")
    
    # Determine the correct script to run based on the variable
    if variable == "sst":
        script_path = "./sst_unet_reorganised.py"
    elif variable == "dsl":
        script_path = "./dsl_unet_reorganised.py"
    else:
        print_error(f"Unknown variable: {variable}")
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
            
            # Replace batch_size value if provided
            if batch_size is not None:
                script_contents = script_contents.replace("batch_size=32", f"batch_size={batch_size}")
                script_contents = script_contents.replace("batch_size=64", f"batch_size={batch_size}")
            
            # Modify to use existing model if required
            if use_existing_model:
                # For UNet, add model loading before the create_model call
                model_creation_line = "model = create_unet_model()"
                var_folder = "sst" if variable == "sst" else "zos"
                model_loading_code = f"""try:
    print('Attempting to load existing UNet model...')
    model = keras.models.load_model('../output/models/unet_{variable}_model.h5', custom_objects={{"mse_loss": custom_mse_loss(mask)}})
    print('Loaded existing model')
except Exception as e:
    print('Creating new UNet model:', e)
    model = create_unet_model()"""
                script_contents = script_contents.replace(model_creation_line, model_loading_code)
            
            # Write modified content to a temporary file
            temp_script_path = f"./temp_{variable}_unet.py"
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
            print_success(f"{variable.upper()} correction completed successfully in {elapsed_time:.2f} seconds")
            return True
        else:
            print_error(f"{variable.upper()} correction failed with return code {return_code}")
            return False
        
    except Exception as e:
        print_error(f"Error executing script: {str(e)}")
        return False

def conduct_ablation_study(variable, epochs=100, batch_sizes=[32, 64, 128]):
    """Conduct an ablation study with different hyperparameters."""
    print_section(f"Conducting Ablation Study for {variable.upper()} using UNet")
    
    results = {}
    
    for batch_size in batch_sizes:
        print_info(f"Testing batch size: {batch_size}")
        success = run_correction(variable, epochs=epochs, batch_size=batch_size)
        results[batch_size] = "Success" if success else "Failed"
    
    # Print summary of ablation study
    print_section("Ablation Study Results")
    for batch_size, result in results.items():
        color = Fore.GREEN if result == "Success" else Fore.RED
        print(f"Batch size {batch_size}: {color}{result}{Style.RESET_ALL}")
    
    return results

def main():
    """Main function to run the climate model error correction toolkit."""
    parser = argparse.ArgumentParser(description="Global Climate Model Error Correction Toolkit")
    parser.add_argument("--variable", type=str, help="Variable to correct (sst, dsl)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--load_model", action="store_true", help="Load an existing model instead of training")
    parser.add_argument("--ablation", action="store_true", help="Conduct ablation study")
    
    args = parser.parse_args()
    
    print_header()
    check_gpu_availability()
    
    variables = list_variables()
    
    # If arguments were provided, use them
    if args.variable:
        if args.variable not in variables:
            print_error(f"Unknown variable: {args.variable}")
            return
        
        if args.ablation:
            check_data_availability(args.variable)
            conduct_ablation_study(args.variable, 
                                  epochs=args.epochs or 100, 
                                  batch_sizes=[args.batch_size] if args.batch_size else [32, 64, 128])
        else:
            check_data_availability(args.variable)
            run_correction(args.variable, args.epochs, args.batch_size, use_existing_model=args.load_model)
    
    # Otherwise, use interactive mode
    else:
        # Get variable choice
        var_choice = None
        while var_choice is None:
            try:
                choice = int(input(f"\n{Fore.BLUE}Select variable to correct (1-{len(variables)}): {Style.RESET_ALL}"))
                if 1 <= choice <= len(variables):
                    var_choice = list(variables.keys())[choice-1]
                else:
                    print_error("Invalid choice. Please try again.")
            except ValueError:
                print_error("Please enter a number.")
        
        # Check data availability with the selected variable
        check_data_availability(var_choice)
        
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
            conduct_ablation_study(var_choice, epochs=epochs)
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
            run_correction(var_choice, epochs, batch_size, use_existing_model=use_existing_model)

if __name__ == "__main__":
    main()