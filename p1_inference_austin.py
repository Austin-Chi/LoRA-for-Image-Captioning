import os
import json
import time
import sys
from datetime import datetime
import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
start = time.time()
formatted_time = datetime.fromtimestamp(start).strftime('%Y-%m-%d %H:%M:%S')

model_id = "llava-hf/llava-1.5-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    quantization_config=bnb_config,
    trust_remote_code=True
).to("cuda")

# Initialize processor and set required attributes
processor = LlavaProcessor.from_pretrained(model_id)
prompt = " Please give a desciption of the image in one sentence."
conversation = f"USER: <image> \n{prompt} ASSISTANT:"
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
generation_config = {
    "max_new_tokens": 60,
    "min_length": 15,
    "num_beams": 3,
    "do_sample": False
}
# Set patch size and vision feature select strategy if they're not already in the processor
if not hasattr(processor, "patch_size"):
    processor.patch_size = 32  # Modify according to model requirements
if not hasattr(processor, "vision_feature_select_strategy"):
    processor.vision_feature_select_strategy = "mean"  # Adjust based on model needs

# Image folder and output file paths
# image_folder = "hw3_data/p1_data/images/val"
# # output_file = "captions_output_austin.json"

def main(image_folder, output_file):
    # Initialize a dictionary to store captions
    captions_dict = {}
    # Loop over each image in the folder
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            image_id = os.path.splitext(filename)[0]  # Remove file extension for image ID

            # Load and process the image
            image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format

            # Create inputs using the simple prompt format
            inputs = processor(text=conversation, images=image, return_tensors="pt").to("cuda", torch.float32)
            # print("inputs:", inputs["input_ids"])

            # Check if pixel values are processed
            if "pixel_values" not in inputs or inputs["pixel_values"] is None:
                print(f"Error: Pixel values not generated correctly for {filename}. Skipping...")
                continue

            # Generate caption
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    **generation_config
                )

            caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract only the assistant's response if needed
            if "ASSISTANT:" in caption:
                caption = caption.split("ASSISTANT:")[-1].strip()  # Get the text after "ASSISTANT:" and strip whitespace
            else:
                print(f"Warning: Expected format not found in caption: {caption}")

            captions_dict[image_id] = caption
            # print(f"Image ID: {image_id}, Caption: {caption}")

    # Save the captions dictionary to a JSON file
    with open(output_file, "w") as f:
        json.dump(captions_dict, f, indent=4)

    print(f"Captions generated and saved to {output_file}")
    end = time.time()
    formatted_time = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
    # print("End time:", formatted_time)
    total_time = end - start
    print("Total time =", total_time)

if __name__ == "__main__":
    image_folder = sys.argv[1]
    output_file = sys.argv[2]
    main(image_folder, output_file)