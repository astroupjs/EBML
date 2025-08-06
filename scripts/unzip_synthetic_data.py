#!/usr/bin/env python3
"""
Universal script to unzip any .tar.gz file if the corresponding folder doesn't exist or is empty.
The extracted files will be saved in a folder with the same name as the input file but without .tar.gz extension.
"""

import os
import tarfile
import glob
import sys
import argparse

def unzip_tar_gz(filename, data_dir="../data"):
    """
    Unzip a .tar.gz file if the corresponding folder doesn't exist or is empty.
    Files will be extracted to a folder with the same name as the input file but without .tar.gz extension.
    
    Args:
        filename (str): Name of the .tar.gz file (e.g., "synthetic_gaia.tar.gz")
        data_dir (str): Directory where the .tar.gz file is located
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Define paths
    tar_file = os.path.join(data_dir, filename)
    
    # Extract folder name from filename (remove .tar.gz extension)
    if filename.endswith('.tar.gz'):
        folder_name = filename[:-7]  # Remove '.tar.gz'
    elif filename.endswith('.tgz'):
        folder_name = filename[:-4]  # Remove '.tgz'
    else:
        print(f"Error: {filename} is not a recognized tar.gz file")
        print("Supported formats: .tar.gz, .tgz")
        return False
    
    # Create target directory path
    target_dir = os.path.join(data_dir, folder_name)
    
    print(f"Input file: {filename}")
    print(f"Target folder: {folder_name}")
    print(f"Full target path: {target_dir}")

    # Check if tar.gz file exists
    if not os.path.exists(tar_file):
        print(f"Error: {tar_file} not found.")
        print(f"Please download the file to {data_dir} directory.")
        return False

    # Check if target directory exists and has files
    should_unzip = False

    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} does not exist. Will unzip {tar_file}")
        should_unzip = True
    else:
        # Check if directory is empty or has no files
        all_files = glob.glob(os.path.join(target_dir, "*"))
        if not all_files:
            print(f"Directory {target_dir} exists but is empty. Will unzip {tar_file}")
            should_unzip = True
        else:
            print(f"Directory {target_dir} exists and contains {len(all_files)} files. Skipping unzip.")
            for file in all_files[:10]:  # Show first 10 files
                print(f"  - {os.path.basename(file)}")
            if len(all_files) > 10:
                print(f"  ... and {len(all_files) - 10} more files")

    # Unzip if needed
    if should_unzip:
        print(f"Unzipping {tar_file} to {target_dir}...")
        try:
            # Create target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=target_dir)
            print("Unzipping completed successfully!")
            
            # Verify the files were extracted
            all_files = glob.glob(os.path.join(target_dir, "*"))
            print(f"Found {len(all_files)} files after extraction:")
            for file in all_files[:10]:  # Show first 10 files
                print(f"  - {os.path.basename(file)}")
            if len(all_files) > 10:
                print(f"  ... and {len(all_files) - 10} more files")
        except Exception as e:
            print(f"Error during unzipping: {e}")
            return False
    else:
        print("No unzipping needed - target folder already exists with files.")
    
    return True

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Unzip a .tar.gz file if needed')
    parser.add_argument('filename', help='Name of the .tar.gz file (e.g., synthetic_gaia.tar.gz)')
    parser.add_argument('--data-dir', default='../data', help='Directory containing the .tar.gz file')
    
    args = parser.parse_args()
    
    success = unzip_tar_gz(args.filename, args.data_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    # If no command line arguments, use default for synthetic_gaia
    if len(sys.argv) == 1:
        success = unzip_tar_gz("synthetic_gaia.tar.gz")
        sys.exit(0 if success else 1)
    else:
        main() 