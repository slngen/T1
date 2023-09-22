from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# 解除内存限制
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Global variables
tile_size = 64
root_dir = '/Code/T1/Dataset/WHU-BCD/Raw'
output_root = f'/Code/T1/Dataset/WHU-BCD/split_{tile_size}'

def save_tile(img_array, path, is_rgb=True):
    if is_rgb:
        im = Image.fromarray(img_array.astype('uint8'), 'RGB')
    else:
        im = Image.fromarray(img_array.astype('uint8'), 'L')
    im.save(path)

def split_image(input_path, output_folder, is_rgb=True):
    img = Image.open(input_path)
    img_array = np.array(img)
    
    y_size, x_size = img_array.shape[0], img_array.shape[1]
    total_tiles = ((y_size + tile_size - 1) // tile_size) * ((x_size + tile_size - 1) // tile_size)
    
    with tqdm(total=total_tiles) as pbar:
        for i in range(0, y_size, tile_size):
            for j in range(0, x_size, tile_size):
                if i + tile_size < y_size:
                    rows = tile_size
                else:
                    rows = y_size - i
                
                if j + tile_size < x_size:
                    cols = tile_size
                else:
                    cols = x_size - j
                
                tile = img_array[i:i+rows, j:j+cols]
                
                # Padding if needed
                padding_dims = ((0, tile_size - tile.shape[0]), (0, tile_size - tile.shape[1])) + ((0, 0),) * (tile.ndim - 2)
                tile = np.pad(tile, padding_dims, 'constant')
                
                tile_path = os.path.join(output_folder, f"x_{j}_y_{i}.tif")
                save_tile(tile, tile_path, is_rgb=is_rgb)
                
                pbar.update(1)

if __name__ == "__main__":
    input_before = os.path.join(root_dir, 'before', 'before.tif')
    input_after = os.path.join(root_dir, 'after', 'after.tif')
    input_label = os.path.join(root_dir, 'change label', 'change_label.tif')

    output_folder_before = os.path.join(output_root, 'A')
    output_folder_after = os.path.join(output_root, 'B')
    output_folder_label = os.path.join(output_root, 'Label')
    
    os.makedirs(output_folder_before, exist_ok=True)
    os.makedirs(output_folder_after, exist_ok=True)
    os.makedirs(output_folder_label, exist_ok=True)

    print("Processing 'before' images...")
    split_image(input_before, output_folder_before, is_rgb=True)
    print("Processing 'after' images...")
    split_image(input_after, output_folder_after, is_rgb=True)
    print("Processing 'label' images...")
    split_image(input_label, output_folder_label, is_rgb=False)

