import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate the YOLO-RL adapter")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the JSON config file")
    parser.add_argument("--device", default=None, help="Override device (cpu, cuda, auto)")
    parser.add_argument("--weights", default=None, help="Override the model weights path")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the batch size")
    parser.add_argument("--img-size", type=int, default=None, help="Override the image size")
    parser.add_argument("--checkpoint-dir", default=None, help="Override the checkpoint directory")
    parser.add_argument("--train", dest="train", action="store_true", help="Force training mode")
    parser.add_argument("--no-train", dest="train", action="store_false", help="Force evaluation-only mode")
    parser.set_defaults(train=None)
    return parser


def load_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    path = Path(config_path or str(DEFAULT_CONFIG_PATH))
    if not path.is_absolute():
        path = ROOT / path

    with path.open("r", encoding="utf-8") as handle:
        config: Dict[str, Any] = json.load(handle)

    if args is not None:
        overrides = {
            "device": getattr(args, "device", None),
            "model_weights": getattr(args, "weights", None),
            "n_epochs": getattr(args, "epochs", None),
            "batch_size": getattr(args, "batch_size", None),
            "img_size": getattr(args, "img_size", None),
            "checkpoint_dir": getattr(args, "checkpoint_dir", None),
            "train": getattr(args, "train", None),
        }
        for key, value in overrides.items():
            if value is not None:
                config[key] = value

    for key in ("checkpoint_dir", "data_dir", "json_path"):
        value = config.get(key)
        if isinstance(value, str):
            path_value = Path(value)
            if not path_value.is_absolute():
                config[key] = str((ROOT / path_value).resolve())

    return config
