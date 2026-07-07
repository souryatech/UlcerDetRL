import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

class HyperKvasirTestDataset(Dataset):
    def __init__(self, img_dir, json_path, img_size=224):
        """
        Args:
            img_dir (str): Path to the folder containing your images
            json_path (str): Path to the bounding_boxes.json file
            img_size (int): Target size to square resize the image (e.g., 224)
        """
        self.img_dir = img_dir
        self.img_size = img_size
        
        # 1. Load the target JSON coordinate file
        with open(json_path, 'r') as f:
            self.bbox_data = json.load(f)
            
        # 2. Filter list down to only files that actively exist in your folder
        self.valid_filenames = [
            fname for fname in self.bbox_data.keys() 
            if os.path.exists(os.path.join(img_dir, fname))
        ]
        
        print(f"Dataset Initialized: {len(self.valid_filenames)} image/box pairs ready for training.")

    def __len__(self):
        return len(self.valid_filenames)

    def __getitem__(self, idx):
        filename = self.valid_filenames[idx]
        img_path = os.path.join(self.img_dir, filename)

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img_resized = TF.resize(img, [self.img_size, self.img_size])
        img_tensor = TF.to_tensor(img_resized)

        boxes_info = self.bbox_data[filename]  # now a LIST of box dicts
        gt_boxes = []
        for box_info in boxes_info:
            xmin, ymin, xmax, ymax = box_info['xmin'], box_info['ymin'], box_info['xmax'], box_info['ymax']
            w_box = xmax - xmin
            h_box = ymax - ymin
            x_center = (xmin + w_box / 2) / orig_w
            y_center = (ymin + h_box / 2) / orig_h
            norm_w = w_box / orig_w
            norm_h = h_box / orig_h
            gt_boxes.append([x_center, y_center, norm_w, norm_h])

        # shape: [N, 4] where N = number of boxes in this image (variable across dataset)
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

        return img_tensor, gt_boxes

def variable_box_collate_fn(batch):
    images, boxes_list = zip(*batch)
    images = torch.stack(images, dim=0)  # images are all same size, fine to stack
    # boxes_list stays a tuple of variable-length [N_i, 4] tensors — don't stack
    return images, list(boxes_list)