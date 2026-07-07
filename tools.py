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

def match_boxes(pred_boxes, gt_boxes):
    """
    Greedy IoU matching between K predicted boxes and N ground-truth boxes.
    pred_boxes, gt_boxes: [K, 4] and [N, 4] in xyxy format (convert first if needed)
    Returns: list of predicted indices, one per gt box, in gt order.
    """
    ious = box_ops.box_iou(pred_boxes, gt_boxes)  # [K, N]
    matched_idx = []
    used = set()
    for n in range(gt_boxes.size(0)):
        order = torch.argsort(ious[:, n], descending=True)
        for idx in order:
            idx = idx.item()
            if idx not in used:
                matched_idx.append(idx)
                used.add(idx)
                break
    return matched_idx