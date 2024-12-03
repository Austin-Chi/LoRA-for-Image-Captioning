# LoRA to finetune a decoder only model for image captioning

A practice on LoRA for Deep Learning and Computer Vision.

## Usage

### Training
[Dataset](https://drive.google.com/file/d/11WqMRxzHcVqvjcbLt61g9Lt1i0fOcgp7/view?usp=sharing)
```bash
python3 P2/P2_lora_train_austin.py
```
modify the paths (see `# Paths to data` in the code)


### Inferencing
Pretrained weight:
```bash
bash hw3_download_ckpt.sh
```

Inference:
```bash
bash bash hw3_2.sh $1 $2 $3
```
$1: path to the folder containing test images (e.g. hw3/p2_data/images/test/)
$2: path to the output json file (e.g. hw3/output_p2/pred.json) 
$3: path to the decoder weights (e.g. hw3/p2_data/decoder_model.bin)

## Reference
[LoRA DLCV](https://docs.google.com/presentation/d/149ViD2g3c0APXAZNGPOFw1ndYAb_UUF19AJZb29uvn4/edit#slide=id.p41)

