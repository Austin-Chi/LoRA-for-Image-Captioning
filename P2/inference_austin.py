import random
import sys
import json
import torch
import os
import math
import collections

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn, Tensor
import torch.nn.functional as F

from PIL import Image
import timm
import timm.data
from tqdm import tqdm

import loralib as lora

from tokenizer import BPETokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
rank = 16

bos_token_id = 50256
padding_idx = -100

prompt = '''Please output ONE sentence that has a subject, a verb, and possibly an object, the environment, and some details. Simply output the one-sentence caption only!! 

Correct examples:
1. "A white sink under a mirror in a bathroom."
2. "The two bears wondering about the point of the camera."
3. "A very large bridge over lots of train tracks."

Now, please provide the descriptive caption of the image provided at the beginning.'''

encoder_file = "encoder.json"
vocab_file = "vocab.bpe"

class EvalDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.images_dir = image_dir
        self.transform = transform
        self.images = []
        self._load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file = self.images[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image_file, image

    def _load_data(self):
        for file in sorted(os.listdir(self.images_dir)):
            self.images.append(file)



class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=rank,lora_alpha=0.5*rank)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=rank,lora_alpha=0.5*rank)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=rank,lora_alpha=0.5*rank)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=rank,lora_alpha=0.5*rank))
        ]))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, r=rank, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, image_features: Tensor = None, max_len: int = None):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        if image_features is not None:
            x = torch.cat([image_features, x], dim=1)
        # Process through transformer
        x = self.transformer.h(x)
        x = self.transformer.ln_f(x)
        
        # Get logits for text positions only
        if image_features is not None:
            x = x[:, -max_len:, :]  # Take only text positions
        x = self.lm_head(x)
        return x

    def generate(self, x: Tensor, image_features: Tensor = None):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        if image_features is not None:
            x = torch.cat([image_features, x], dim=1)
        
        # Process through transformer
        x = self.transformer.h(x)
        x = self.transformer.ln_f(x)
        
        # Take last position for next token prediction
        x = x[:, -1, :]  # This will be the last text position
        x = self.lm_head(x)
        return x

class ImageCaptioningModel(nn.Module):
    def __init__(self, decoder_path, vit_encoder_model, tokenizer, prompt):
        super().__init__()
        cfg = Config(checkpoint=decoder_path)
        self.decoder = Decoder(cfg).to(device)
        self.vit_encoder_model = vit_encoder_model
        self.tokenizer = tokenizer
        self.projector = nn.Linear(1024, 768)
        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
        self.loss = 0
        self.image_prompt = prompt#open(image_prompt_path, "r").read()
        # bos_token_id = 50256#self.tokenizer.encode(BOS_TOKEN, allowed_special=BOS_TOKEN)[
        #     0
        # ]

    def forward(self, images, captions=None, max_new_tokens=30, mode="train"):
        if mode == "train":
            image_features = self.vit_encoder_model.forward_features(images)
            image_features = self.projector(image_features)
            img_seq_len = image_features.size(1)

            image_prompt_tokens = self.tokenizer.encode(self.image_prompt)
            image_prompt_len = len(image_prompt_tokens)

            text_tokens = []
            target_tokens = []
            for caption in captions:
                encoded_cap = self.tokenizer.encode(caption)
                if encoded_cap[0] != bos_token_id:
                    encoded_cap = [bos_token_id] + encoded_cap
                if encoded_cap[-1] != bos_token_id:
                    encoded_cap += [bos_token_id]
                text_tokens.append(encoded_cap[:-1])  # Remove last token
                target_tokens.append(encoded_cap[1:])  # Remove first token

            max_len = max(len(t) for t in text_tokens)
            text_tokens = [image_prompt_tokens + self._pad_text_tokens(t, max_len, bos_token_id) for t in text_tokens]
            target_tokens = [self._pad_text_tokens(t, max_len, padding_idx) for t in target_tokens]

            text_tokens = torch.tensor(text_tokens).to(device)
            target_tokens = torch.tensor(target_tokens).to(device)

            # Get target tokens from captions
            outputs = self.decoder(text_tokens, image_features, max_len)
            # print(text_tokens.shape, target_tokens.shape, outputs.shape)

            return outputs, target_tokens
        elif mode == "eval":
            batch_size = len(images)

            # Get image features
            image_features = self.vit_encoder_model.forward_features(images)
            image_features = self.projector(image_features)

            image_prompt_tokens = self.tokenizer.encode(self.image_prompt)
            generated_tokens = torch.tensor(image_prompt_tokens, device=device)
            generated_tokens = generated_tokens.repeat(batch_size, 1)
            # Initialize with start tokens
            bos_tokens = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=device
            )
            generated_tokens = torch.cat([generated_tokens, bos_tokens], dim=1)
            for _ in range(max_new_tokens):
                next_token_logits = self.decoder.generate(
                    generated_tokens, image_features
                ).log_softmax(-1)
                next_tokens = next_token_logits.argmax(-1, keepdim=True)

                # Construct next input
                generated_tokens = torch.cat([generated_tokens, next_tokens], dim=-1)

            generated_tokens = generated_tokens.tolist()

            first_eos_idx = generated_tokens[0].index(bos_token_id)
            for i in range(batch_size):
                try:
                    generated_tokens[i] = generated_tokens[i][first_eos_idx+1:generated_tokens[i].index(bos_token_id, first_eos_idx+1)]
                except:
                    generated_tokens[i] = generated_tokens[i][first_eos_idx+1:]
            return generated_tokens

    def _pad_text_tokens(self, text_token, max_text_seq_len, pad_token):
        text_token = text_token + [pad_token] * (max_text_seq_len - len(text_token))
        return text_token


def main(image_folder, output_json_path, decoder_model_path, checkpoint_path):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    tokenizer = BPETokenizer(encoder_file=encoder_file, vocab_file=vocab_file)
    vit_encoder_model = timm.create_model("vit_large_patch14_clip_224.laion2b", pretrained=True)
    data_config = timm.data.resolve_model_data_config(vit_encoder_model)
    image_transform = timm.data.create_transform(**data_config, is_training=False)

    model = ImageCaptioningModel(
        decoder_model_path, vit_encoder_model, tokenizer, prompt
    ).to(device)

    # try:
    model.load_state_dict(
        torch.load(checkpoint_path),
        strict=False,
    )
    # calculate total parameters in the checkpoint path only
    checkpoint_params = sum(p.numel() for p in torch.load(checkpoint_path).values())
    print(f"Total parameters: {checkpoint_params}")
    # except Exception as e:
    #     print(f"Cannot load checkpoint, starting from scratch")

    val_dataset = EvalDataset(image_folder, image_transform)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    outputs = {}
    model.eval()
    with torch.no_grad():
        for image_filenames, images in tqdm(
            val_dataloader, desc="Validation"
        ):
            images = images.to(device)
            # predicted_tokens = model.generate(images, max_new_tokens=30)
            predicted_tokens = model(images, max_new_tokens=30, mode="eval")
            decoded_captions = tokenizer.batch_decode(predicted_tokens)
            # print('decoded_captions: ', decoded_captions)
            assert len(image_filenames) == len(decoded_captions)
            for image_filename, decoded_caption in zip(
                image_filenames, decoded_captions
            ):
                image_filename = image_filename.split(".")[0]
                outputs[image_filename] = decoded_caption
            # break
    
    with open(output_json_path, "w") as f:
        json.dump(outputs, f, indent=4)

if __name__ == "__main__":
    ckpt_path = "P2/output_p3/checkpoints/model_lora_r16_59000_lation.pt"
    main(sys.argv[1], sys.argv[2], sys.argv[3], ckpt_path)
