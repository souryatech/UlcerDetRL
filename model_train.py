import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from huggingface_hub import login

from config import build_arg_parser, load_config
from dataloader import HyperKvasirTestDataset
from eval import evaluate_model
from foundation_model_reward import gemma_model_reward
from models import YOLO_RL_Adapter, load_medgemma_critic, load_yolo_model


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config, args)

    if not config.get("train", True):
        print("Training disabled in config; exiting.")
        return

    if config.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])

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
    pytorch_yolo = load_yolo_model(model_weights)
    model = YOLO_RL_Adapter(pytorch_yolo).to(device)
    print("Detection model loaded successfully")

    print("Loading medgemma model...")
    login()
    reward_model, reward_processor = load_medgemma_critic(device=device)

    model.train()
    print("Loading Dataset..")

    ds = HyperKvasirTestDataset(img_dir=data_dir, json_path=json_path, img_size=img_size)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(
        [
            {"params": model.box_mean_head.parameters()},
            {"params": model.box_std_head.parameters()},
        ],
        lr=learning_rate,
    )

    sup_loss_func = nn.MSELoss()

    for epoch in range(n_epochs):
        print(f"\n--- Starting Epoch {epoch} ---")

        for batch_idx, (images, gt_boxes) in enumerate(data_loader):
            images = images.to(device)
            gt_boxes = gt_boxes.to(device)

            optimizer.zero_grad()
            means, stds = model(images)

            sup_loss = sup_loss_func(means, gt_boxes)

            coord_distribution = Normal(means, stds)
            sampled_boxes = coord_distribution.sample()
            log_probs = coord_distribution.log_prob(sampled_boxes).sum(dim=-1)

            rewards = []
            for i in range(images.size(0)):
                reward = gemma_model_reward(images[i], sampled_boxes[i], reward_model, reward_processor)
                rewards.append(reward)
            rewards = torch.stack(rewards).to(images.device)

            if rewards.size(0) > 1:
                base_reward = rewards.mean()
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
