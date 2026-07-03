import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from config import build_arg_parser, load_config
from dataloader import HyperKvasirTestDataset
from models import YOLO_RL_Adapter, load_yolo_model
from tools import xywh_to_xyxy


def evaluate_model(val_loader, device, current=False, model=None, weights_path=None):
    print("Loading model for evaluation...")
    if not current:
        if weights_path is None:
            raise ValueError("weights_path is required when current=False")
        raw_pytorch_yolo = load_yolo_model(weights_path)
        model = YOLO_RL_Adapter(raw_pytorch_yolo).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))

    if model is None:
        raise ValueError("A model instance is required for evaluation")

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    print("\n--- Starting Evaluation ---")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            images = batch_data[0].to(device)
            gt_boxes = batch_data[1].to(device)

            predicted_means, _ = model(images)
            pred_boxes_xyxy = xywh_to_xyxy(predicted_means)
            gt_boxes_xyxy = xywh_to_xyxy(gt_boxes)

            preds = []
            targets = []

            for i in range(len(images)):
                pred_box = pred_boxes_xyxy[i].unsqueeze(0)
                gt_box = gt_boxes_xyxy[i].unsqueeze(0)

                preds.append(
                    {
                        "boxes": pred_box,
                        "scores": torch.tensor([1.0], device=device),
                        "labels": torch.tensor([0], device=device),
                    }
                )
                targets.append({"boxes": gt_box, "labels": torch.tensor([0], device=device)})

            metric.update(preds, targets)
            print(f"\r  Evaluating Batch {batch_idx + 1}/{len(val_loader)}", end="")

    print("\n\nCalculating final metrics (this may take a few seconds)...")
    results = metric.compute()

    print("\n--- Final Results ---")
    print(f"mAP (IoU 0.50:0.95): {results['map'].item():.4f}")
    print(f"AP50 (IoU 0.50)    : {results['map_50'].item():.4f}")
    print(f"APm (Medium objects) : {results['map_medium'].item():.4f}")
    print(f"APl (Large objects)  : {results['map_large'].item():.4f}")
    return results


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config, args)

    if config.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])

    batch_size = int(config.get("batch_size", 1))
    img_size = int(config.get("img_size", 224))
    data_dir = config.get("data_dir", "hyper_kvasir_mock/images")
    json_path = config.get("json_path", "hyper_kvasir_mock/bounding_boxes.json")
    weights_path = config.get("model_weights")

    if not weights_path:
        raise ValueError("Provide a model_weights entry in config.json or use --weights")

    if not os.path.isabs(weights_path):
        weights_path = str((Path(__file__).resolve().parent / weights_path).resolve())

    ds = HyperKvasirTestDataset(img_dir=data_dir, json_path=json_path, img_size=img_size)
    val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    evaluate_model(val_loader, device, current=False, weights_path=weights_path)


if __name__ == "__main__":
    main()
