import os
import re
import shutil

# Path to the logs directory
logs_dir = "/unity/f2/asugandhi/DIS/MonsoonForecast/logs"

# Create a directory to store runs if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# Regex pattern to extract the last digits from the file name
pattern = re.compile(r'events\.out\.tfevents\.\d+\.(\d+)\.\d+$')


# Iterate through files in the logs directory
for filename in os.listdir(logs_dir):
    # Match the pattern to extract the last digits
    match = pattern.match(filename)
    if match:
        last_digits = match.group(1)
        
        # Create a directory for the last digits if it doesn't exist
        run_dir = os.path.join(logs_dir, last_digits)
        os.makedirs(run_dir, exist_ok=True)
        
        # Move the file to the corresponding run directory
        src_path = os.path.join(logs_dir, filename)
        dest_path = os.path.join(run_dir, filename)
        shutil.move(src_path, dest_path)
        print(f"Moved {filename} to {dest_path}")
