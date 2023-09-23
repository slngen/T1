import os
import shutil
import random
from tqdm import tqdm

# Global Variables
root_folder = '/Code/T1/Dataset/WHU-BCD/split_64'  # Change to your root folder path
ratio = 0.7  # 70% for training, 30% for evaluation
categories = ['A', 'B', 'Label']  # Categories or sub-folders you have

def split_dataset(root_folder, categories):
    # Initialize tqdm progress bar
    src_folder = os.path.join(root_folder, categories[0])
    filenames = os.listdir(src_folder)
    pbar = tqdm(total=len(filenames), desc=f"Processing {src_folder.split('/')[-1]}")
    
    # Shuffle and split filenames
    random.shuffle(filenames)
    split_idx = int(len(filenames) * ratio)
    train_files = filenames[:split_idx]
    eval_files = filenames[split_idx:]
    
    for category in categories:
        src_folder = os.path.join(root_folder, category)
        train_folder = os.path.join(root_folder, 'Train', category)
        eval_folder = os.path.join(root_folder, 'Eval', category)

        # Create train and eval subfolders for each category
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(eval_folder, exist_ok=True)

        # Move files
        for filename in train_files:
            shutil.move(os.path.join(src_folder, filename), os.path.join(train_folder, filename))

        for filename in eval_files:
            shutil.move(os.path.join(src_folder, filename), os.path.join(eval_folder, filename))

        pbar.update(len(train_files) + len(eval_files))
    
    pbar.close()

if __name__ == "__main__":
    # Split and move the files
    split_dataset(root_folder, categories)
