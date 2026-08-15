import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def gemma_model_reward(
    image,
    sampled_box,
    medgemma_model,
    processor,
    min_size: int = 16,
    negative_penalty: float = -1.0,
):
    """
    Safely crop `image` using `sampled_box` (xywh normalized) and query the VLM.
    Returns a scalar tensor on the same device as `image`.

    If the box is invalid or smaller than `min_size` pixels in width/height,
    bypass the VLM and return `negative_penalty` as a tensor.
    """
    with torch.no_grad():
        device = image.device
        _, H, W = image.shape

        # Ensure sampled_box is on CPU/float for numeric ops
        sb = sampled_box.detach().cpu().float()
        x_c, y_c, w, h = float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3])

        # Quick validity checks
        if not (math.isfinite(x_c) and math.isfinite(y_c) and math.isfinite(w) and math.isfinite(h)):
            return torch.tensor(negative_penalty, device=device)

        # Compute pixel coordinates (float)
        xmin_f = (x_c - w / 2.0) * W
        ymin_f = (y_c - h / 2.0) * H
        xmax_f = (x_c + w / 2.0) * W
        ymax_f = (y_c + h / 2.0) * H

        # Clamp to image boundaries
        xmin_f = max(0.0, min(xmin_f, float(W)))
        ymin_f = max(0.0, min(ymin_f, float(H)))
        xmax_f = max(0.0, min(xmax_f, float(W)))
        ymax_f = max(0.0, min(ymax_f, float(H)))

        # Convert to integer pixel indices (left, top, right, bottom)
        left = int(math.floor(xmin_f))
        top = int(math.floor(ymin_f))
        right = int(math.ceil(xmax_f))
        bottom = int(math.ceil(ymax_f))

        # Ensure valid ordering
        if right <= left or bottom <= top:
            return torch.tensor(negative_penalty, device=device)

        crop_w = right - left
        crop_h = bottom - top

        # Enforce minimum physical size
        if crop_w < int(min_size) or crop_h < int(min_size):
            return torch.tensor(negative_penalty, device=device)

        # Perform the crop (PIL conversion expects CPU tensor)
        try:
            cropped_img = TF.crop(image, top, left, crop_h, crop_w)
            pil_crop = TF.to_pil_image(cropped_img.cpu())
        except Exception:
            return torch.tensor(negative_penalty, device=device)

        # Build prompt and move inputs to the model device
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Does this image show a colorectal polyp? Answer strictly with one word, Yes or No:",
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        try:
            inputs = processor(text=prompt, images=pil_crop, return_tensors="pt")
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            yes_token_id = processor.tokenizer.convert_tokens_to_ids("Yes")
            no_token_id = processor.tokenizer.convert_tokens_to_ids("No")

            outputs = medgemma_model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            target_logits = torch.stack([next_token_logits[yes_token_id], next_token_logits[no_token_id]])
            probabilities = F.softmax(target_logits, dim=0)
            reward = probabilities[0]
            return reward.detach().to(device)
        except Exception:
            return torch.tensor(negative_penalty, device=device)
