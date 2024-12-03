#!/bin/bash

python -c "import clip; clip.load('ViT-B/32')" && echo "CLIP model ViT-B/32 loaded successfully!"
python -c "import timm; timm.create_model('vit_large_patch14_clip_224.laion2b', pretrained=True)" && echo "ViT-Large model loaded successfully!"
wget -O ./p2_model.pt 'https://www.dropbox.com/scl/fi/pgjbh3wx02dxu5jkxfp6k/p2_model.pt?rlkey=o9sxc995kkzumawg6dkzuko4g&st=307al4f1&dl=1'
