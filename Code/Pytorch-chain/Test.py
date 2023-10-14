import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from Config import config
from Backbone import Backbone
'''
Network
'''
ckpt_path = r"/Code/T1/Models/2023-10-08_06-27--DiceL2-32x64tox64/L2-32-E290-0.9174.ckpt"
model = Backbone()
net_state = torch.load(ckpt_path)
model.load_state_dict(net_state)
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

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

    return image, binary_label

def slide_window(img, patch_size=64, stride=32):
    patches = []
    img_array = np.array(img)

    height, width, _ = img_array.shape

    for i in range(0, height, stride):
        for j in range(0, width, stride):
            patch_array = img_array[i:i+patch_size, j:j+patch_size]

            # Padding directly in numpy array if the patch size is less than required
            if patch_array.shape[0] < patch_size or patch_array.shape[1] < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                padded_patch[:patch_array.shape[0], :patch_array.shape[1]] = patch_array
                patches.append(transform(padded_patch))
            else:
                patches.append(transform(patch_array))

    return torch.stack(patches)

def inference_on_patches(model, patches):
    with torch.no_grad():
        preds = []
        for patch in patches:
            output = model(patch.unsqueeze(0))
            pred = nn.Softmax(dim=1)(output)  # Assuming your model does not apply softmax
            preds.append(pred.squeeze(0))
        return torch.stack(preds)
    
def reconstruct_from_patches(patches, img_shape, patch_size=64, stride=32):
    recon = np.zeros((*img_shape, 2))
    count = np.zeros(img_shape)
    idx = 0
    for i in range(0, img_shape[0], stride):
        for j in range(0, img_shape[1], stride):
            h_end = min(i+patch_size, img_shape[0])
            w_end = min(j+patch_size, img_shape[1])
            
            recon[i:h_end, j:w_end] += patches[idx].numpy()[:h_end-i, :w_end-j]
            count[i:h_end, j:w_end] += 1
            idx += 1
    return recon / count[..., np.newaxis]

image, binary_label = load_image_pair_and_label("after/before")
patches = slide_window(image.permute(1, 2, 0))  # Switch CxHxW to HxWxC

predicted_patches = inference_on_patches(model, patches)
reconstructed_image = reconstruct_from_patches(predicted_patches, image.shape[1:])

# To get the predicted class (either 0 or 1)
predicted_class = np.argmax(reconstructed_image, axis=-1)
# Calculate confusion matrix, F1 score
confusion = confusion_matrix(binary_label.ravel(), predicted_class.ravel())
f1 = f1_score(binary_label.ravel(), predicted_class.ravel())

# Save predicted image
pred_img = Image.fromarray((predicted_class * 255).astype(np.uint8))
pred_img.save("predicted.png")

print("Confusion Matrix:", confusion)
print("F1 Score:", f1)