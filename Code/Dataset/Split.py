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

def save_tile(img_array, path):
    img_array = img_array.astype('uint8')
    # Check if the image is single channel (2D array)
    if len(img_array.shape) == 2:
        img_array *= 255
        img_array = np.stack([img_array] * 3, axis=2)  # Convert to shape (H, W, 3)
    print(f"Label image shape: {img_array.shape}, dtype: {img_array.dtype}")
    im = Image.fromarray(img_array, 'RGB')
    im.save(path)

def split_image(input_paths, output_folders):
    images = [Image.open(path) for path in input_paths]
    img_arrays = [np.array(img) for img in images]
    
    y_size, x_size = img_arrays[0].shape[0], img_arrays[0].shape[1]
    total_tiles = ((y_size + tile_size - 1) // tile_size) * ((x_size + tile_size - 1) // tile_size)
    
    with tqdm(total=total_tiles) as pbar:
        for i in range(0, y_size, tile_size):
            for j in range(0, x_size, tile_size):
                tiles = [img_array[i:i+tile_size, j:j+tile_size] for img_array in img_arrays]

                # Check if the label tile is all black
                if np.all(tiles[-1] == 0):
                    pbar.update(1)
                    continue

                for tile, output_folder in zip(tiles, output_folders):
                    if i + tile_size < y_size:
                        rows = tile_size
                    else:
                        rows = y_size - i
                    if j + tile_size < x_size:
                        cols = tile_size
                    else:
                        cols = x_size - j

                    # Padding if needed
                    padding_dims = ((0, tile_size - rows), (0, tile_size - cols)) + ((0, 0),) * (tile.ndim - 2)
                    tile = np.pad(tile, padding_dims, 'constant')
                    
                    tile_path = os.path.join(output_folder, f"x_{j}_y_{i}.png")
                    save_tile(tile, tile_path)
                
                pbar.update(1)

if __name__ == "__main__":
    input_paths = [
        os.path.join(root_dir, 'before', 'before.tif'),
        os.path.join(root_dir, 'after', 'after.tif'),
        os.path.join(root_dir, 'change label', 'change_label.tif')
    ]

    output_folders = [
        os.path.join(output_root, 'A'),
        os.path.join(output_root, 'B'),
        os.path.join(output_root, 'Label')
    ]

    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    print("Processing images...")
    split_image(input_paths, output_folders)
