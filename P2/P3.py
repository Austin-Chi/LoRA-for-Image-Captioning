import argparse
import json
import os
import pathlib
import random

import torch
import timm
import timm.data
from PIL import Image
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt

from model import ImageCaptionModel
from tokenizer import BPETokenizer
from config import (
    DEVICE,
    BATCH_SIZE,
    DATA_CFG,
    SEED,
)

image_prompt_path = DATA_CFG["image_prompt_path"]
encoder_file = DATA_CFG["encoder_file"]
vocab_file = DATA_CFG["vocab_file"]

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

class Image_dataset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()

        self.Transform = transform
        self.Image_names = [p for p in root.glob("*")]

    def __getitem__(self, idx):
        img = Image.open(self.Image_names[idx]).convert('RGB')
        img = self.Transform(img)

        return img, os.path.splitext(os.path.basename(self.Image_names[idx]))[0]

    def __len__(self):
        return len(self.Image_names)


# def register_attention_hook(model, features, feature_sizes):
#     def hook_decoder(module, ins, outs):
#         features.append(outs.clone().detach().cpu())
#     handle_decoder = model.decoder.transformer.h[-1].attn.register_forward_hook(
#         hook_decoder)
#     print("Hook registered")
#     print('feature size:', len(features))
#     print('feature size:', features[0].size())
#     return [handle_decoder]
def register_attention_hook(model, features, feature_sizes):
    """
    Registers a forward hook to capture attention weights for visualization.
    
    Args:
        model: The ImageCaptionModel or its decoder.
        features: A list to store the extracted features.
        feature_sizes: A list to store the sizes of the extracted features.
        
    Returns:
        A list of hook handles that can be removed later.
    """
    handles = []
    # Define a hook function to capture the attention weights
    def hook_fn(module, input, output):
        # Extract the attention scores from the `att` matrix inside the `Attention` module
        B, T, C = input[0].size()  # input[0] is the input tensor to the attention module
        att_weights = module.att  # Shape: [batch_size, num_heads, seq_len, seq_len]
        # avg_attention = att_weights.mean(dim=1)  # Average across heads: [batch_size, seq_len, seq_len]
        avg_attention = att_weights[:, 5, :, :]  # Attention of the ith head: [batch_size, seq_len, seq_len]
        features.append(avg_attention.detach().cpu())  # Detach and move to CPU
        feature_sizes.append((T, T))  # Save feature sizes (token sequence length)
        print('feature size:', len(features))
        print('feature size:', features[-1].size())

    # Register the hook on the last attention layer of the decoder
    # last_attention_layer = model.decoder.transformer.h[0].attn
    last_attention_layer = model.decoder.transformer.h[1].attn
    # last_attention_layer = model.decoder.transformer.h[-10].attn
    handle = last_attention_layer.register_forward_hook(hook_fn)
    handles.append(handle)
    
    return handles


def vis_atten_map(atten_mat, ids, first_word_ind, feature_size, image_fn, image_path, tokenizer):
    print(atten_mat.shape)
    print('ids: ', len(ids))
    print('first_word_ind: ', first_word_ind)
    nrows = len(ids) // 5 if len(ids) % 5 == 0 else len(ids) // 5 + 1
    ncols = 5
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 8))
    feature_size = (16, 16)  # H/patch size, W/patch size = feature size
    for i, id in enumerate(ids):
        attn_vector = atten_mat[i + 257 + first_word_ind, 1:257]
        attn_map = torch.reshape(attn_vector, feature_size)
        attn_map -= torch.min(attn_map)
        attn_map /= torch.max(attn_map)
        # print(torch.min(attn_map), torch.max(attn_map))
        # print(attn_map.size())
        im = Image.open(image_path)
        size = im.size
        mask = resize(attn_map.unsqueeze(0), [size[1], size[0]]).squeeze(0)
        mask = np.uint8(mask * 255)
        # print(mask.shape)
        ax[i // 5][i % 5].imshow(im)
        if i == 0:
            ax[i // 5][i % 5].set_title('<SOS>')
        elif i == len(ids) - 1:
            ax[i // 5][i % 5].set_title('<EOS>')
            ax[i // 5][i % 5].imshow(mask, alpha=0.7, cmap='jet')
        else:
            ax[i // 5][i % 5].set_title(tokenizer.decode([id]))
            ax[i // 5][i % 5].imshow(mask, alpha=0.7, cmap='jet')
        ax[i // 5][i % 5].axis('off')
    for i in range(len(ids), nrows * ncols):
        ax[i // 5][i % 5].axis('off')
    plt.savefig(args.output_dir / image_fn)


def main(args):
    decoder_model_path, checkpoint_path = args.decoder_path, args.ckpt_path
    # transform = transforms.Compose([
    #     transforms.Resize(
    #         224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[
    #                          0.2290, 0.2240, 0.2250])
    # ])
    
    seed_everything(SEED)
    tokenizer = BPETokenizer(encoder_file=encoder_file, vocab_file=vocab_file)
    encoder = timm.create_model("vit_large_patch14_clip_224.laion2b", pretrained=True)
    data_config = timm.data.resolve_model_data_config(encoder)
    validation_preprocess = timm.data.create_transform(**data_config, is_training=False)
    valid_set = Image_dataset(
        root=args.image_dir,
        transform=validation_preprocess,
    )
    Model = ImageCaptionModel(
        decoder_model_path, encoder, tokenizer, image_prompt_path
    ).to(DEVICE)

    try:
        Model.load_state_dict(
            torch.load(checkpoint_path),
            strict=False,
        )
        # calculate total parameters in the checkpoint path only
        checkpoint_params = sum(p.numel() for p in torch.load(checkpoint_path).values())
        print(f"Total parameters in checkpoint: {checkpoint_params}")
    except Exception as e:
        print(f"Cannot load checkpoint, starting from scratch")


    Model.eval()

    for data, name in valid_set:
        features, feature_sizes = [], []
        to_rm_l = register_attention_hook(Model, features, feature_sizes)
        images = data.unsqueeze(0).to(DEVICE)
        output_ids, first_word_ind = Model.generate_for_viz(images, max_new_tokens=30)
        print('output_ids: ', output_ids)
        output_tokens = tokenizer.batch_decode(output_ids)#['<SOS>']
        print('output_tokens: ', output_tokens)
        # output_tokens.extend([tokenizer.id_to_token(id) for id in output_ids])
        # output_tokens.append('<EOS>')

        # visualize
        # Retrieve the last attention map
        attention_matrix = features[-1].squeeze(0)  # Shape: [seq_len, seq_len]
        print('attention_matrix shape: ', attention_matrix.size())

        # Visualize attention
        vis_atten_map(
            atten_mat=attention_matrix,
            ids=output_ids[0],
            first_word_ind=first_word_ind,
            feature_size=feature_sizes,
            image_fn=name,
            image_path=(args.image_dir / name).with_suffix('.jpg'),
            tokenizer=tokenizer,
        )
        for handle in to_rm_l:
            handle.remove()


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--image_dir', type=pathlib.Path,
                        default='../hw3_data/p3_data/images/')#'../hw3_data/p3_data/images/'
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--decoder_path",
                        type=pathlib.Path, default="../hw3_data/p2_data/decoder_model.bin")

    parser.add_argument("--ckpt_path",
                        type=pathlib.Path, default="output_p3/checkpoints/model_lora_r16_59000_lation.pt")

    # Validation args
    parser.add_argument("--output_dir", type=pathlib.Path, default="P3_plot_austin_val")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    main(args)