
import kagglehub
import shutil
import os

# 1. Download the latest version to the kagglehub cache
print("Downloading dataset...")
downloaded_path = kagglehub.dataset_download("nadaibrahim/coco2014")
print("Downloaded to cache at:", downloaded_path)

# 2. Define your destination (current folder)
destination_path = os.path.join(os.getcwd(), "data/coco2014")

# 3. Move the files
if not os.path.exists(destination_path):
    print(f"Moving dataset to {destination_path}...")
    shutil.move(downloaded_path, destination_path)
    print("Move complete!")
else:
    print(f"Folder '{destination_path}' already exists. Skipping move.")

# 4. Verify the contents
print("Current directory contents:", os.listdir(destination_path))