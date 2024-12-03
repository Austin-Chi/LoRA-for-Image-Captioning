
import os
import json
import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import time
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 設置隨機種子來保證一致性
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Model and processor setup
model_id = "llava-hf/llava-1.5-7b-hf"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float32,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to("cuda")

# Initialize processor
processor = LlavaProcessor.from_pretrained(model_id)

# Set patch size and vision feature select strategy if they're not already in the processor
if not hasattr(processor, "patch_size"):
    processor.patch_size = 32
if not hasattr(processor, "vision_feature_select_strategy"):
    processor.vision_feature_select_strategy = "mean"

image_folder = sys.argv[1]
output_file = sys.argv[2]
layer = sys.argv[3]
head = sys.argv[4]

# 如果輸出文件不存在，創建一個空的 JSON 文件
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        json.dump({}, f)
    print(f"Created an empty JSON file at {output_file}")

prompt = " Please give a desciption of the image in one sentence."
conversation = f"USER: <image> \n{prompt} ASSISTANT:"
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
generation_config = {
    "max_new_tokens": 60,
    "min_length": 15,
    "num_beams": 3,
    "do_sample": False,
    "output_attentions": True,
    "return_dict_in_generate": True
}
captions_dict = {}

# Visualize attention maps
def visualize_attention_map(attentions, image_path, output_path, tokens):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(15, 10))
    vision_token_start = len(processor.tokenizer(conversation.split("<image>")[0], return_tensors='pt')["input_ids"][0]) + 1
    vision_token_end = vision_token_start + 576  # Adjust for your model's grid size
    # print("vision_token_start: ", vision_token_start)
    # print("vision_token_end: ", vision_token_end)
    joint_attentions = None
    for i, token in enumerate(tokens):
        if i >= 10:  # Limit to first 10 tokens for visualization
            break
        per_token_attention = attentions[i]
        last_layer_attention = per_token_attention[layer]
        print("last_layer_attention shape: ", last_layer_attention.size())
        if False:#i==0:
            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            attention = last_layer_attention[0, :, :, :]
            print("attention shape: ", attention.size())
            att_mat = torch.mean(attention, dim=0)
            print("att_mat shape: ", att_mat.size())
            residual_att = torch.eye(att_mat.size(1)).to(att_mat.device)
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
            joint_attentions = aug_att_mat
            attention_vec = joint_attentions[0, vision_token_start:vision_token_end]
        else:
            attention = last_layer_attention[0, :, -1, vision_token_start:vision_token_end]  # Get attention for the <image> token
            # att_mat = torch.mean(attention, dim=0).squeeze(0)
            att_mat = attention[head].squeeze(0)
            att_mat = att_mat / att_mat.sum(dim=-1)
            # matrix multiplication to get the attention of the image tokens
            # joint_attentions = joint_attentions.to(dtype=torch.float32)
            # att_mat = att_mat.to(dtype=torch.float32)
            attention_vec = att_mat#joint_attentions @ att_mat
        # attention.shape = [32, 576], 32 is the number of heads, 576 is the number of tokens in the image
        # scale each head's attention to [0, 1]
        # print("attentions shape for: ", attention.shape)
        # average_attention = attention.mean(dim=0).squeeze()  # Averages over dim=1 and removes dimensions of size 1
        attn_map = attention_vec.reshape(24, 24)# Reshape to [24, 24]
        attn_map = attn_map.cpu().numpy()
        attn_map -= attn_map.min()
        attn_map /= attn_map.max()

        # Resize to original image size
        attn_map_resized = np.array(Image.fromarray(np.uint8(attn_map * 255)).resize(image.size, resample=Image.BICUBIC))
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.imshow(attn_map_resized, cmap='jet', alpha=0.5)
        plt.title(f'Token: {token}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Loop over each image in the folder
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        image_id = os.path.splitext(filename)[0]

        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=conversation, images=image, return_tensors="pt").to("cuda", torch.float32)

        if "pixel_values" not in inputs or inputs["pixel_values"] is None:
            print(f"Error: Pixel values not generated correctly for {filename}. Skipping...")
            continue

        # Generate caption with attention
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            print("inputs: ", inputs["input_ids"].size(), inputs["pixel_values"].size())
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                **generation_config
            )
            # print("outputs: ", outputs.sequences[0].size())
            #print the attributes of the outputs
            print(outputs.__doc__)
            for attr in dir(outputs):
                if attr == "__doc__":
                    print(outputs.__doc__)
            # 0/0

        caption = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        print("caption: ", caption)
        # 提取注意力权重并可视化
        attentions = outputs.attentions
        # turn the attention into a tensor
        # attentions = torch.stack(attentions).squeeze(1)
        print("attentions length", len(outputs.attentions))
        if attentions is not None:
            tokens = processor.tokenizer.convert_ids_to_tokens(outputs.sequences[0][inputs["input_ids"].size(1):])
            output_path = os.path.join('p3_austin', f"{image_id}_attention.png")
            visualize_attention_map(attentions, image_path, output_path, tokens)

        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1].strip()

        captions_dict[image_id] = caption
         # 释放内存
# Save the captions dictionary to a JSON file
with open(output_file, "w") as f:
    json.dump(captions_dict, f, indent=4)

print(f"Captions and attention maps generated and saved.")
