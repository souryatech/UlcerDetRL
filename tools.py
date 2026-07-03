import torch
import torchvision.ops.boxes as box_ops

def xywh_to_xyxy(boxes):
    """Converts [x_c, y_c, w, h] format to [x_min, y_min, x_max, y_max]"""
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x_min = x_c - (w / 2)
    y_min = y_c - (h / 2)
    x_max = x_c + (w / 2)
    y_max = y_c + (h / 2)
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)