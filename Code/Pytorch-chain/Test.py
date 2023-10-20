import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from Config import config
from Backbone import Backbone
Image.MAX_IMAGE_PIXELS = None

stride = 64
patch_size = 64

# Network loading
ckpt_path = r"/Code/T1/Models/2023-10-17_05-34--DiceAL2-16x64tox64aug/AL2-16-E640-0.9175.ckpt"
model = Backbone()
net_state = torch.load(ckpt_path)
model.load_state_dict(net_state)
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

def get_confusion_matrix_elements(gt, pred):
    TP = np.sum((gt == 1) & (pred == 1))
    TN = np.sum((gt == 0) & (pred == 0))
    FP = np.sum((gt == 0) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))
    return TP, TN, FP, FN

def compute_f1(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def load_image_pair_and_label(root_dir):
    path_A = f"{root_dir}/before/before.tif"
    path_B = f"{root_dir}/after/after.tif"
    path_label = f"{root_dir}/change label/change_label.tif"
    
    image_A = transform(Image.open(path_A).convert('RGB'))
    image_B = transform(Image.open(path_B).convert('RGB'))
    image = torch.cat([image_A, image_B], dim=0)

    label = transform(Image.open(path_label).convert('RGB'))
    white_mask = (label[0] > 0.9) & (label[1] > 0.9) & (label[2] > 0.9)
    binary_label = white_mask.float().numpy()

    # image = image[:,64*30:64*50,64*30:64*50]
    # binary_label = binary_label[64*30:64*50,64*30:64*50]
    return image, binary_label

def slide_window(img):
    patches = []
    img_array = np.array(img)

    height, width, _ = img_array.shape

    for i in range(0, height, stride):
        for j in range(0, width, stride):
            patch_array = img_array[i:i+patch_size, j:j+patch_size]

            # Padding directly in numpy array if the patch size is less than required
            if patch_array.shape[0] < patch_size or patch_array.shape[1] < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 6), dtype=np.uint8)
                padded_patch[:patch_array.shape[0], :patch_array.shape[1]] = patch_array
                patches.append(transform(padded_patch))
            else:
                patches.append(transform(patch_array))

    return torch.stack(patches)

def inference_on_patches(model, patches):
    model.to(config.device)
    with torch.no_grad():
        preds = []
        # Split patches into batches based on the specified batch size from config
        num_batches = len(patches) // config.batch_size + int(len(patches) % config.batch_size > 0)
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * config.batch_size
            end_idx = start_idx + config.batch_size
            batch = patches[start_idx:end_idx]
            batch = batch.to(config.device)
            # Since model already applies softmax, we don't need to apply it again
            output = model(batch)[0].cpu()
            preds.extend(output)
        return torch.stack(preds)

    
def reconstruct_from_patches(patches, img_shape):
    patches = patches.permute(0, 2, 3, 1)
    recon = np.zeros((*img_shape, 2))
    idx = 0
    for i in range(0, img_shape[0], stride):
        for j in range(0, img_shape[1], stride):
            h_end = min(i+patch_size, img_shape[0])
            w_end = min(j+patch_size, img_shape[1])
            
            recon[i:h_end, j:w_end] += patches[idx].numpy()[:h_end-i, :w_end-j]
            idx += 1
    return recon

image, binary_label = load_image_pair_and_label("/Code/T1/Dataset/WHU-BCD/Raw")
patches = slide_window(image.permute(1, 2, 0))  # Switch CxHxW to HxWxC

predicted_patches = inference_on_patches(model, patches)
reconstructed_image = reconstruct_from_patches(predicted_patches, image.shape[1:])

# To get the predicted class (either 0 or 1)
predicted_class = np.argmax(reconstructed_image, axis=-1)
TP, TN, FP, FN = get_confusion_matrix_elements(binary_label.ravel(), predicted_class.ravel())
f1 = compute_f1(TP, FP, FN)
acc = (binary_label.ravel()*predicted_class.ravel()).sum() / predicted_class.ravel().shape[0]

# Displaying results
print("True Positives:", TP)
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)
print("F1 Score:", f1)
print("Acc:", acc)

# Save predicted image
pred_img = Image.fromarray((predicted_class * 255).astype(np.uint8))
pred_img.save("predicted-s{}.png".format(stride))

# Calculate difference
difference_image = np.abs(binary_label - predicted_class) * 255  # Multiply by 255 to get white where there's a difference

# Save difference image
diff_img = Image.fromarray(difference_image.astype(np.uint8))
diff_img.save("diff-s{}.png".format(stride))

# Save label image
label_img = Image.fromarray((binary_label * 255).astype(np.uint8))
label_img.save("label.png".format(stride))