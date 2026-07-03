import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def gemma_model_reward(image, sampled_box, medgemma_model, processor):
    with torch.no_grad():
        device = image.device
        _, H, W = image.shape

        x_c, y_c, w, h = sampled_box
        xmin = int((x_c - w / 2) * W)
        ymin = int((y_c - h / 2) * H)
        xmax = int((x_c + w / 2) * W)
        ymax = int((y_c + h / 2) * H)

        xmin = max(0, min(xmin, W - 1))
        ymin = max(0, min(ymin, H - 1))
        xmax = max(xmin + 1, min(xmax, W))
        ymax = max(ymin + 1, min(ymax, H))

        crop_h = ymax - ymin
        crop_w = xmax - xmin
        cropped_img = TF.crop(image, ymin, xmin, crop_h, crop_w)

        pil_crop = TF.to_pil_image(cropped_img.cpu())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Does this endoscopic tissue crop contain a mucosal ulcer? Answer strictly with one word, Yes or No:"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=pil_crop, return_tensors="pt").to(device)
        yes_token_id = processor.tokenizer.convert_tokens_to_ids("Yes")
        no_token_id = processor.tokenizer.convert_tokens_to_ids("No")

        outputs = medgemma_model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        target_logits = torch.stack([next_token_logits[yes_token_id], next_token_logits[no_token_id]])
        probabilities = F.softmax(target_logits, dim=0)
        reward = probabilities[0]

    return reward.detach()
