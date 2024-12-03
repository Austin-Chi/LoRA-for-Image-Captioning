import argparse
import csv
import json
import os
import pathlib

import clip
import torch
from torch import Tensor, nn
from PIL import Image
from tqdm.auto import tqdm
import logging
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tokenizer import BPETokenizer
import timm
import torchvision.transforms as trns
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from transformers import BitsAndBytesConfig
import loralib as lora
import math
import collections
import logging

# Set up logging configuration
log_file = "p2_training_austin.log"
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()         # Also log to console
    ]
)

# logging.basicConfig(
#     filename="my_log.log",
#     level=logging.INFO,
#     format="%(message)s",
# )

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"device = {device}")
PAD_TOKEN = 50256
UNK_TOKEN = 1
BOS_TOKEN = 50256
EOS_TOKEN = 50256
rank = 8
lora_alpha = 0.5*rank

# Define start and end token IDs based on the tokenizer
vit_model_name = 'vit_large_patch14_clip_224.laion2b'#Laion2b
tokenizer = BPETokenizer(encoder_file="encoder.json",  vocab_file="vocab.bpe")
endoftext_token = "<|endoftext|>"
endoftext_token_id = 50256
padding_token_id = 0  # Assuming padding token ID is 0

def custom_collate_fn(batch):
    images, captions = zip(*batch)
    
    # Add start and end tokens to each caption
    encoded_captions = [
        torch.cat([torch.tensor([endoftext_token_id]), caption, torch.tensor([endoftext_token_id])], dim=0)
        for caption in captions
    ]
    
    # Pad captions
    padded_captions = pad_sequence(encoded_captions, batch_first=True, padding_value=padding_token_id)
    
    # Stack images
    images = torch.stack(images)
    
    return images, padded_captions

# image_transforms = transforms.Compose([
#     # transforms.Resize((224, 224)),  # Resize to match ViT's expected input size
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization values for ImageNet
# ])
image_transforms = create_transform(
    **resolve_data_config({}, model=vit_model_name)
)

class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, annotations_file, tokenizer, transform=None):
        """
        Args:
            image_folder (str): Path to the image directory.
            annotations_file (str): Path to the JSON annotations file.
            tokenizer (BPETokenizer): The tokenizer for encoding captions.
            transform (callable, optional): Transform to be applied on the images.
        """
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        # Map image IDs to file names
        self.image_id_to_file = {img['id']: img['file_name'] for img in data['images']}
        # Create a list of (image_id, caption) pairs
        self.captions = [(ann['image_id'], ann['caption']) for ann in data['annotations']]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id, caption = self.captions[idx]
        image_file = self.image_id_to_file[image_id]
        image_path = os.path.join(self.image_folder, image_file)
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        caption_tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.long)
        
        return image, caption_tokens


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
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=rank,lora_alpha=lora_alpha)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=rank,lora_alpha=lora_alpha)
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
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=rank,lora_alpha=lora_alpha)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=rank,lora_alpha=lora_alpha))
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
            wte = lora.Embedding(cfg.vocab_size, cfg.n_embd, r=rank,lora_alpha=lora_alpha),
            wpe = lora.Embedding(cfg.block_size, cfg.n_embd, r=rank,lora_alpha=lora_alpha),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False, r=rank,lora_alpha=lora_alpha)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    # def forward(self, x: Tensor):
    #     x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
    #     pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
    #     x = self.transformer.wte(x) + self.transformer.wpe(pos)
    #     x = self.lm_head(self.transformer.ln_f(self.transformer.h(x)))
    #     return x
    def forward(self, input_embeddings, caption_tokens=None):
        # Use position embeddings
        B, T, _ = input_embeddings.size()
        pos = torch.arange(T, dtype=torch.long, device=input_embeddings.device).unsqueeze(0)
        pos_embeddings = self.transformer.wpe(pos).expand(B, -1, -1)
        
        # Combine input_embeddings with position embeddings
        x = input_embeddings + pos_embeddings

        if caption_tokens is not None:
            # Get token embeddings for the caption tokens (teacher forcing)
            token_embeddings = self.transformer.wte(caption_tokens)
            # Create positional encodings for caption tokens starting after input embeddings
            caption_start_pos = T  # Start position for captions after input embeddings
            caption_pos = torch.arange(caption_start_pos, caption_start_pos + token_embeddings.size(1), dtype=torch.long, device=token_embeddings.device).unsqueeze(0)
            caption_pos_embeddings = self.transformer.wpe(caption_pos).expand(B, -1, -1)

            # Add positional encoding to caption token embeddings
            token_embeddings = token_embeddings + caption_pos_embeddings

            x = torch.cat((x, token_embeddings), dim=1)

        x = self.transformer.ln_f(self.transformer.h(x))
        return self.lm_head(x)


class ImageCaptioningModel(nn.Module):
    def __init__(self, cfg, tokenizer, vit_model_name=vit_model_name):
        super().__init__()
        # Initialize Vision Transformer Encoder (ViT)
        self.vit_encoder = timm.create_model(
            vit_model_name, pretrained=True, num_classes=0,
        ).to(device)
        # Ensure the ViT encoder does not include the classification head
        self.vit_encoder.reset_classifier(0)  # Remove classifier if present
        self.decoder = Decoder(cfg).to(device)
        self.tokenizer = tokenizer

        # Linear layer to map ViT output to match decoder's embedding size
        self.proj = nn.Linear(self.vit_encoder.embed_dim, cfg.n_embd).to(device)

    def top_p_sampling(self, logits, top_p=0.9, temperature=0.6, repetition_penalty=1.2, past_tokens=None):
        """
        Apply Top-p (nucleus) sampling with temperature and repetition penalty.
        
        Args:
            logits (Tensor): Logits of shape [batch_size, vocab_size] for the current token.
            top_p (float): Cumulative probability threshold for nucleus sampling.
            temperature (float): Temperature scaling factor for logits.
            repetition_penalty (float): Penalty factor for repeated tokens.
            past_tokens (Tensor): List of previously generated tokens for repetition penalty.

        Returns:
            Tensor: Selected token IDs of shape [batch_size].
        """
        with torch.no_grad():
            # Apply repetition penalty
            if past_tokens is not None:
                for batch_idx, tokens in enumerate(past_tokens):
                    for token in tokens:
                        # Penalize logits of previously generated tokens
                        if logits[batch_idx, token] < 0:
                            logits[batch_idx, token] *= repetition_penalty
                        else:
                            logits[batch_idx, token] /= repetition_penalty

            # Apply temperature scaling
            logits = logits / temperature

            # Compute softmax probabilities
            probs = F.softmax(logits, dim=-1)

            # Initialize list for selected tokens
            next_tokens = []

            # Process each item in the batch independently
            for i in range(probs.size(0)):
                sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)

                # Mask out indices where cumulative probability exceeds top_p
                sorted_indices_to_keep = cumulative_probs < top_p
                sorted_indices_to_keep[0] = True  # Always keep at least one token

                # Get the filtered probabilities and renormalize
                filtered_indices = sorted_indices[sorted_indices_to_keep]
                filtered_probs = probs[i][filtered_indices]
                filtered_probs = filtered_probs / filtered_probs.sum()

                # Sample from the filtered probabilities
                next_token = filtered_indices[torch.multinomial(filtered_probs, 1)]
                next_tokens.append(next_token)

        return torch.stack(next_tokens)


    def forward(self, image, prompt=None, caption_tokens=None, mode='train', max_length=30, p=0.9, temperature=0.3, repetition_penalty=1.2):
        # Step 1: Pass the image through the ViT encoder
        # vit_outputs = self.vit_encoder(image)
        start_token_idx = 257 - 1
        vit_outputs = self.vit_encoder.forward_features(image)
        # print("vit output shape: ", vit_outputs.size())
        # Project the ViT output to match the decoder's embedding size
        image_embedding = self.proj(vit_outputs)  # Take the [CLS] token representation
        gen_len = caption_tokens.size(1)+1 if caption_tokens is not None else 0
        # print("gen_len: ", gen_len)
        if prompt is not None:
            # Step 2: Tokenize the prompt and get embeddings
            prompt_tokens = torch.tensor(self.tokenizer.encode(prompt), device=image.device).unsqueeze(0)
            prompt_embeddings = self.decoder.transformer.wte(prompt_tokens)
            # Duplicate the prompt embeddings for each item in the batch
            prompt_embeddings = prompt_embeddings.repeat(image.size(0), 1, 1)  # Repeat along batch dimension


        # Step 3: Concatenate prompt embeddings with the image embedding
        # image_embedding = image_embedding.unsqueeze(1)  # Make it a single "token"
        if prompt is not None:
            input_embeddings = torch.cat((prompt_embeddings, image_embedding), dim=1)
            start_token_idx = input_embeddings.size(1) - 1
            # print("start_token_idx: ", start_token_idx)
        else:
            input_embeddings = image_embedding

        # Training Mode (Teacher Forcing)
        if mode == 'train' and caption_tokens is not None:
            # print("input embeddings shape: ", input_embeddings.size())
            output = self.decoder(input_embeddings, caption_tokens)[:, start_token_idx:, :]
            # print("output shape: ", output.size())
            return output  # Return logits for loss calculation
        # elif mode == 'train':
        #     current_input = input_embeddings
        #     # Store past tokens to apply repetition penalty
        #     past_tokens = [[] for _ in range(image.size(0))]
        #     generated_logits = None
        #     for l in range(gen_len):
        #         output = self.decoder(current_input)  # Pass the sequence so far to the decoder
        #         logits = output[:, -1, :]  # Get logits for the last token in the sequence
        #         # print("logits: ", logits.size())
        #         if l == gen_len - 1:
        #             generated_logits = output[:, 256:, :]
        #             # print("generated_logits: ", generated_logits.size())
        #             return generated_logits
        #         # Use Top-p sampling with temperature and repetition penalty
        #         next_token = self.top_p_sampling(logits, top_p=p, temperature=temperature, repetition_penalty=repetition_penalty, past_tokens=past_tokens)
        #         # print("next_token: ", next_token)
        #         # Append the generated token to the sequence
        #         # generated_logits = torch.cat((generated_logits, logits.unsqueeze(1)), dim=1)  # Shape [batch_size, sequence_length, vocab_size]
        #         # print("logits: ", generated_logits.size())
        #         # Update past tokens for repetition penalty
        #         for i, token in enumerate(next_token):
        #             past_tokens[i].append(token.item())

        #         # # Check if all sequences have generated the end token
        #         # if (next_token == endoftext_token_id).all():
        #         #     break

        #         # Get embedding for the next token and append to current input
        #         next_token_embedding = self.decoder.transformer.wte(next_token)#.unsqueeze(1)
        #         # print("next_token_embedding: ", next_token_embedding.size())
        #         # print("current_input: ", current_input.size())
        #         current_input = torch.cat((current_input, next_token_embedding), dim=1)

        #     # Concatenate all generated tokens along the sequence dimension
        #     return generated_logits  # Shape: [batch_size, sequence_length, vocab_size]

        # if mode == 'train' and caption_tokens is not None:
        #     output_predictions = []

        #     # Start with the initial input_embeddings (prompt + image token)
        #     current_input = input_embeddings  # Shape: [batch_size, prompt_length + 1, cfg.n_embd]

        #     for i in range(caption_tokens.size(1)):
        #         # Step 4: Pass the current input sequence to the decoder
        #         next_token_logits = self.decoder(current_input)  # Shape: [batch_size, seq_len, vocab_size]
                
        #         output_predictions.append(next_token_logits.unsqueeze(1))  # Add dimension for seq_len

        #         # Step 5: Teacher forcing - use the ground truth token as the next input
        #         next_token = caption_tokens[:, i]  # Shape: [batch_size]
        #         next_token_embedding = self.decoder.transformer.wte(next_token).unsqueeze(1)  # Shape: [batch_size, 1, cfg.n_embd]

        #         # Append the next token to the current input sequence
        #         current_input = torch.cat((current_input, next_token_embedding), dim=1)

        #     # Concatenate all predictions along the sequence length
        #     outputs = torch.cat(output_predictions, dim=1)  # Shape: [batch_size, seq_len, vocab_size]
        #     return outputs  # Return logits for loss calculation

        # Inference Mode (Autoregressive Decoding)
        elif mode == 'inference':
            generated_tokens = []
            current_input = input_embeddings
            # Store past tokens to apply repetition penalty
            past_tokens = [[] for _ in range(image.size(0))]

            for _ in range(max_length):
                output = self.decoder(current_input)  # Pass the sequence so far to the decoder
                logits = output[:, -1, :]  # Get logits for the last token in the sequence
                
                # Use Top-p sampling with temperature and repetition penalty
                next_token = self.top_p_sampling(logits, top_p=p, temperature=temperature, repetition_penalty=repetition_penalty, past_tokens=past_tokens)
                # print("next_token: ", next_token)
                # Append the generated token to the sequence
                generated_tokens.append(next_token.unsqueeze(1))  # Shape [batch_size, 1]

                # Update past tokens for repetition penalty
                for i, token in enumerate(next_token):
                    past_tokens[i].append(token.item())

                # # Check if all sequences have generated the end token
                # if (next_token == endoftext_token_id).all():
                #     break

                # Get embedding for the next token and append to current input
                next_token_embedding = self.decoder.transformer.wte(next_token)#.unsqueeze(1)
                # print("next_token_embedding: ", next_token_embedding.size())
                # print("current_input: ", current_input.size())
                current_input = torch.cat((current_input, next_token_embedding), dim=1)

            # Concatenate all generated tokens along the sequence dimension
            return torch.cat(generated_tokens, dim=1)  # Shape: [batch_size, sequence_length]

def norm_long(x):
    x /= x.norm(dim=-1, keepdim=True)
    return x.long()

def train_model(model, dataloader, val_dataloader, optimizer, scheduler, num_epochs, device, prompt=None, save_folder='p2_outputs_austin'):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=padding_token_id)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
            images, captions = batch
            images, captions = images.to(device), captions.to(device)

            # Tokenize the prompt once per batch
            input_prompt = prompt

            # Shift target captions for teacher forcing
            input_captions = captions[:, :-1]
            target_captions = captions[:, :]

            # Forward pass with teacher forcing
            outputs = model(image=images, prompt=input_prompt, caption_tokens=input_captions, mode='train')
            # outputs = model(image=images, prompt=input_prompt, caption_tokens=None, mode='train')
            # print("output shape: ", outputs.size())
            # Calculate the difference in length between outputs and target_captions
            # output_len = outputs.size(1)
            # target_len = target_captions.size(1)
            # pad_len = output_len - target_len

            # # Pad target captions in the front to match output length
            # if pad_len > 0:
            #     target_captions = F.pad(target_captions, (pad_len, 0), value=padding_token_id)

            # outputs = outputs.view(-1, outputs.size(-1))
            outputs = outputs.reshape(-1, outputs.size(-1))  # Use .reshape() instead of .view()
            # print("target captions shape: ", target_captions.size())
            target_captions = target_captions.contiguous().view(-1)
            # print("target captions shape: ", target_captions.size())
            # print("outputs shape: ", outputs.size())
            

            # Compute loss and backpropagate
            loss = criterion(outputs, target_captions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Visualization: Decode and print the last batch's output and target captions
            if batch_idx == len(dataloader) - 1:  # Last batch of the epoch
                with torch.no_grad():  # Disable gradient tracking for visualization
                    # Reshape to original batch format for decoding
                    output_tokens = model(image=images, prompt=input_prompt, mode='inference').view(images.size(0), -1)
                    target_tokens = target_captions.view(images.size(0), -1)
                    
                    for i in range(min(3, images.size(0))):  # Show a few examples from the last batch
                        generated_caption = tokenizer.decode(output_tokens[i].cpu().numpy())
                        target_caption = tokenizer.decode(target_tokens[i].cpu().numpy())
                        logging.info(f"\nGenerated Caption: {generated_caption}")
                        logging.info(f"Target Caption: {target_caption}\n")
        # do validation
        for batch_idx, batch in enumerate((val_dataloader)):
            if batch_idx > 0:
                break
            images, captions = batch
            images, captions = images.to(device), captions.to(device)

            # Tokenize the prompt once per batch
            input_prompt = prompt

            # Shift target captions for
            input_captions = captions[:, :-1]
            target_captions = captions[:, :]
            # Forward pass with teacher
            with torch.no_grad():  # Disable gradient tracking for visualization
                # Reshape to original batch format for decoding
                output_tokens = model(image=images, prompt=input_prompt, mode='inference').view(images.size(0), -1)
                target_tokens = target_captions.view(images.size(0), -1)
                
                for i in range(min(3, images.size(0))):  # Show a few examples from the last batch
                    generated_caption = tokenizer.decode(output_tokens[i].cpu().numpy())
                    target_caption = tokenizer.decode(target_tokens[i].cpu().numpy())
                    logging.info(f"\nGenerated Caption: {generated_caption}")
                    logging.info(f"Target Caption: {target_caption}\n")

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        trainable_weights = [
            name for name, param in model.named_parameters() if param.requires_grad == True
        ]
        save_weights = {
            k: v for k, v in model.state_dict().items() if k in trainable_weights
        }
        torch.save(save_weights, os.path.join(save_folder, f"model_lora_r{rank}_{epoch}_1113_768_austin.pt"))



def main():
    # args parameters
    EPOCHS = 5
    logging.info("end of text id: ", endoftext_token_id)
    # Dataloader setting
    # 根據timm model config 去設定transform條件
    
    # Paths to data
    image_folder = 'hw3_data/p2_data/images/train/'
    annotations_file = 'hw3_data/p2_data/train.json'
    val_image_folder = 'hw3_data/p2_data/images/val/'
    val_annotations_file = 'hw3_data/p2_data/val.json'
    save_folder = 'p2_outputs_austin_multi_tokens_teacher_forcing_with_prompt_laion2b'
    os.makedirs(save_folder, exist_ok=True)

    prompt = '''Please output ONE sentence that has a subject, a verb, and possibly an object, the environment, and some details. Simply output the one-sentence caption only!! 

Correct examples:
1. "A white sink under a mirror in a bathroom."
2. "The two bears wondering about the point of the camera."
3. "A very large bridge over lots of train tracks."

Now, please provide the descriptive caption of the image provided at the beginning.'''

    # Create the dataset and dataloader
    dataset = ImageCaptionDataset(
        image_folder=image_folder,
        annotations_file=annotations_file,
        tokenizer=tokenizer,
        transform=image_transforms
    )

    val_dataset = ImageCaptionDataset(
        image_folder=val_image_folder,
        annotations_file=val_annotations_file,
        tokenizer=tokenizer,
        transform=image_transforms
    )

    # Create DataLoader
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    # Model
    cfg = Config("hw3_data/p2_data/decoder_model.bin")
    #pretrained checkpoint
    checkpoint_path = "p2_outputs_austin_multi_tokens_teacher_forcing_with_prompt_laion2b/model_lora_r8_11_1113_768_austin.pt"
    lora_cfg = torch.load(checkpoint_path)
    model = ImageCaptioningModel(cfg=cfg, tokenizer=tokenizer, vit_model_name=vit_model_name).to(device)
    model.load_state_dict(lora_cfg, strict=False)
    lora.mark_only_lora_as_trainable(model)
    # Freeze encoder
    for param in model.proj.parameters():
        param.requires_grad = True

    logging.info(
        f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M"
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}")

    

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # scheduler = lr_scheduler.LinearLR(
    #     optimizer=optimizer,  # Your optimizer here
    #     start_factor=1.0,  # Start with the full learning rate
    #     end_factor=0.0,    # Linearly reduce to zero
    #     total_iters=EPOCHS  # Number of epochs for linear decay
    # )
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=EPOCHS, T_mult=2, eta_min=1e-6
    )

    # Run the training loop
    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs=EPOCHS, device=device, prompt=prompt, save_folder=save_folder)# prompt right now is useless




if __name__ == "__main__":
    main()
