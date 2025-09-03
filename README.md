

# Symbolic Graphics Programming with Large Language Models







## Installation


```bash
conda env create -n sgp_gen -f environment.yml

conda activate sgp_gen

pip install vllm==0.7.2 && pip install oat-llm==0.0.9

git clone git@github.com:Sphere-AI-Lab/SGP-RL.git

cd SGP-RL

pip install -e .

pip install cairosvg openai-clip lxml
```
We provide a requirements.txt recording the version informations for reference.


## Quick Start

### Prepare Training Datasets


1) Setup COCO 2017 dataset(assume you put it in COCO_DIR):
```bash
export COCO_DIR=YOUR_COCO_DIR
cd "$COCO_DIR"
# Images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Captions annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

```
You should have:
- train2017/
- val2017/
- annotations/
    - captions_train2017.json
    - captions_val2017.json
    ...

2) setup the svg training data:
download the dataset file at https://huggingface.co/datasets/haoquan03/SVG-Gen-70k/blob/main/svg-gen-70k.jsonl

Put it into YOUR_SVG_DIR and setup environment variables:
```bash
export SVG_DIR=YOUR_SVG_DIR
```


### Prepare SGP-Single-9k Datasets (for evaluation)
Download SGP-Single-9k dataset at  
https://huggingface.co/datasets/haoquan03/SGP-Single-9k/blob/main/eval.json
https://huggingface.co/datasets/haoquan03/SGP-Single-9k/blob/main/train.json

Put it into YOUR_SVG_DIR.


### RL Training

```bash
bash train_zero_svg.sh
```

### Evaluation on SGP-GenBench
Sampling model responses and calculating DINO-score, CLIP-score and Diversity on SGP-:
```bash
python evaluate_svg_model.py 
    --model_path YOUR_MODEL_PATH  
```
To get the VQA and HPS metrics, check [eval_tools](eval_tools/)

Evaluation on SGP-CompBench:
[sgp-compbench](sgp-compbench/)




## Citation

```python
PLACE HOLDER
```

