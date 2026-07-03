# UlcerDetRL

Minimal repository-style setup for training and evaluating the YOLO-RL adapter.

## Quick start

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
   On Windows PowerShell, use:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Review the defaults in config.json.
4. Run training or evaluation:
   ```bash
   python model_train.py --config config.json --train
   python eval.py --config config.json
   ```

   To evaluate a specific checkpoint or weights file, pass it explicitly:
   ```bash
   python eval.py --config config.json --weights checkpoints/yolo_rl_epoch_3.pt
   ```

## Configuration

The project uses config.json plus CLI overrides. Common options include:
- train: enable training mode
- device: cpu, cuda, or auto
- model_weights: YOLO weights file
- n_epochs, batch_size, img_size
- checkpoint_dir, data_dir, json_path
