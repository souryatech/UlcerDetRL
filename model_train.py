import os
from pathlib import Path
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from huggingface_hub import login

from config import build_arg_parser, get_device, load_config
from dataloader import HyperKvasirTestDataset, variable_box_collate_fn
from eval import evaluate_model
from foundation_model_reward import gemma_model_reward
from models import YOLO_RL_Adapter, load_medgemma_critic, load_yolo_model
from tools import xywh_to_xyxy, match_boxes  


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config, args)

    if not config.get("train", True):
        print("Training disabled in config; exiting.")
        return

    device = get_device(config.get("device", "auto"))

    learning_rate = float(config.get("learning_rate", 0.001))
    alpha = float(config.get("alpha", 1.0))
    beta = float(config.get("beta", 0.5))
    n_epochs = int(config.get("n_epochs", 10))
    checkpoint_interval = int(config.get("checkpoint", 3))
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    batch_size = int(config.get("batch_size", 1))
    img_size = int(config.get("img_size", 224))
    model_weights = config.get("model_weights", "yolo11n.pt")
    data_dir = config.get("data_dir", "hyper_kvasir_mock/images")
    json_path = config.get("json_path", "hyper_kvasir_mock/bounding_boxes.json")

    if not os.path.isabs(model_weights):
        model_weights = str((Path(__file__).resolve().parent / model_weights).resolve())

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("Loading detection model...")
    pytorch_yolo = load_yolo_model(model_weights, device=device)
    model = YOLO_RL_Adapter(pytorch_yolo).to(device)
    print("Detection model loaded successfully")

    print("Loading medgemma model...")
    login()
    reward_model, reward_processor = load_medgemma_critic(device=device)

    model.train()
    print("Loading Dataset..")

    ds = HyperKvasirTestDataset(img_dir=data_dir, json_path=json_path, img_size=img_size)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=variable_box_collate_fn)

    optimizer = optim.Adam(
        [
            {"params": model.box_mean_head.parameters()},
            {"params": model.box_std_head.parameters()},
        ],
        lr=learning_rate,
    )

    sup_loss_func = nn.MSELoss()

    # Estimate training time by running one warmup training step (optional).
    if config.get("estimate_time", True):
        try:
            print("Estimating training time via one warmup step...")
            model.train()
            it = iter(data_loader)
            images_w, gt_boxes_w = next(it)
            images_w = images_w.to(device)
            gt_boxes_w = [g.to(device) for g in gt_boxes_w]

            optimizer.zero_grad()
            start = time.perf_counter()
            means_w, stds_w = model(images_w)
            # use a simple surrogate loss for timing (no expensive reward call)
            dummy_target = torch.zeros_like(means_w)
            sup_loss_est = sup_loss_func(means_w, dummy_target)
            sup_loss_est.backward()
            optimizer.step()
            elapsed = time.perf_counter() - start

            num_samples = len(ds)
            steps_per_epoch = math.ceil(num_samples / batch_size)
            total_steps = steps_per_epoch * n_epochs
            est_total_seconds = elapsed * total_steps
            est_minutes = est_total_seconds / 60.0
            est_hours = est_minutes / 60.0
            print(
                f"Warmup step time: {elapsed:.3f}s — Estimated total: {est_hours:.2f}h ({est_minutes:.1f}min) over {total_steps} steps"
            )
        except Exception as e:
            print(f"Time estimation skipped: {e}")

    for epoch in range(n_epochs):
        print(f"\n--- Starting Epoch {epoch} ---")

        for batch_idx, (images, gt_boxes) in enumerate(data_loader):
            images = images.to(device)
            gt_boxes = [g.to(device) for g in gt_boxes]

            optimizer.zero_grad()
            means, stds = model(images)

            sup_losses = []
            for b in range(images.size(0)):
                gt_b = gt_boxes[b]                     # [N_i, 4], xywh
                if gt_b.size(0) == 0:
                    continue
                pred_xyxy = xywh_to_xyxy(means[b])          # [K, 4]
                gt_xyxy = xywh_to_xyxy(gt_b)                 # [N_i, 4]
                matched_idx = match_boxes(pred_xyxy, gt_xyxy)

                matched_preds = means[b][matched_idx]        # [N_i, 4], matched xywh preds
                sup_losses.append(sup_loss_func(matched_preds, gt_b))

            sup_loss = torch.stack(sup_losses).mean()

            coord_distribution = Normal(means, stds)
            sampled_boxes = coord_distribution.sample()
            log_probs = coord_distribution.log_prob(sampled_boxes).sum(dim=-1)

            rewards = torch.zeros(images.size(0), 3, device=device)
            for i in range(images.size(0)):
                for k in range(3):
                    rewards[i, k] = gemma_model_reward(
                        images[i], sampled_boxes[i, k], reward_model, reward_processor
                    )
            flat_rewards = rewards.view(-1)
            if flat_rewards.numel() > 1:
                base_reward = flat_rewards.mean()
                adjusted_rewards = rewards - base_reward
            else:
                # If batch size is 1, just use the raw reward directly
                adjusted_rewards = rewards
            loss_rl = -(log_probs * adjusted_rewards).mean()
            total_loss = (alpha * sup_loss) + (beta * loss_rl)

            total_loss.backward()
            optimizer.step()

            print(
                f"\r  Batch {batch_idx + 1}/{len(data_loader)} | Total Loss: {total_loss.item():.3f} | Sup: {sup_loss.item():.3f} | RL: {loss_rl.item():.3f}",
                end="",
            )

        if epoch % checkpoint_interval == 0 and epoch != 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"yolo_rl_epoch_{epoch}.pt")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "supervised_loss": sup_loss.item(),
                "rl_loss": loss_rl.item(),
                "total_loss": total_loss.item(),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f" Saved periodic checkpoint to {checkpoint_path}")

    print("Training Complete, Evaluating Model..")
    evaluate_model(data_loader, device, model=model, current=True)


if __name__ == "__main__":
    main()
