# Evaluation Tools

This directory contains tools for evaluating SVG generation models using VQA scores (from [TIFA](https://github.com/Yushi-Hu/tifa/)) and HPSv2 metrics.

## Prerequisites

Additional dependencies for evaluation:
```bash
pip install hpsv2 
```

## VQA Evaluation

### 1. Build VQA Evaluation Data

#### For Custom Datasets

Use `get_vqa_tifa.py` to generate VQA question-answer pairs for your dataset:

```bash
python build_vqa/get_vqa_tifa.py \
    --input_file path/to/your/dataset.json \
    --output_file path/to/output/questions.json \
    --dataset [coco|sgp_compbench|vgbench] \
    --batch_size 1 \
    --rank 0 \
    --world_size 1
```

**Parameters:**
- `--input_file`: Path to your dataset JSON file
- `--output_file`: Output path for generated questions
- `--dataset`: Dataset type (coco, sgp_compbench, or vgbench)
- `--batch_size`: Batch size for question generation (default: 1)
- `--rank`: Process rank for distributed processing (default: 0)
- `--world_size`: Total number of processes (default: 1)

### 2. Convert VQA Annotations

Convert the generated questions to the evaluation format:

```bash
python build_vqa/convert_vqa_annos.py \
    --input_files path/to/questions.json \
    --dataset [coco|sgp_compbench|vgbench] \
    --caption_file path/to/captions.json \
    --output_file path/to/output/vqa_eval.json
```

**Parameters:**
- `--input_files`: Path(s) to question files (can specify multiple)
- `--dataset`: Dataset type
- `--caption_file`: Caption file (required for COCO dataset)
- `--output_file`: Output path for converted annotations

#### For Pre-built Datasets

VQA data for evaluation on COCO, SG-Compbench, and VGBench have been prepared in `build_vqa/vqa_data/`:
- `coco_qa.json`: COCO dataset VQA questions
- `sgp_compbench_qa.json`: SG-Compbench dataset VQA questions  
- `vgbench_qa.json`: VGBench dataset VQA questions

### 3. Run VQA Evaluation

#### Step 1: Generate ID to Image Mapping

Create a mapping between sample IDs and generated image file names:

```bash
python get_id2img.py \
    --result_dir path/to/model/results \
    --dataset [coco|sgp_compbench|vgbench] \
    --caption_file path/to/captions.json \
    --output_file path/to/output/id2img.json
```

**Parameters:**
- `--result_dir`: Directory containing model generation results
- `--dataset`: Dataset type
- `--caption_file`: Caption file for the dataset
- `--output_file`: Output path for ID-to-image mapping (default: `result_dir/id2img.json`)

#### Step 2: Run VQA Score Evaluation

```bash
python eval_vqa_tifa.py \
   --model_name mplug-large \
   --question_answer_path path/to/dataset_qa.json \
   --result_dir path/to/model/results
```
## HPSv2 Evaluation

Evaluate human preference scores for generated images:

```bash
python eval_hpsv2.py \
    --input_dir path/to/model/results \
    --output_file path/to/output/hpsv2_results.pkl
```

**Output:**
- Results saved as pickle file with per-image scores and statistics
- Text file with mean HPSv2 score for the entire dataset
