from typing import Dict, List, Optional, Union, Any
import os
import json
from datasets import Dataset, IterableDataset, DatasetDict
import torch
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path


image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think>\n<answer> answer here </answer>"
)

class COCOImageDataset:
    """
    Dataset loader and processor for COCO Image Dataset from local files
    """
    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        instruction_prompt: str = "Please write SVG code for generating the image corresponding to the following description:",
        think_tokens: bool = True,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Load the dataset from local files at ~/data/coco
        """
        # Read from env; fall back to ~/data/coco
        _COCO_ENV = os.getenv("COCO_DIR", "~/data/coco")

        # Expand ~ and any $VARS the user might put in the env value
        coco_dir = Path(os.path.expandvars(os.path.expanduser(_COCO_ENV)))

        # (optional) sanity check
        if not coco_dir.exists():
            raise FileNotFoundError(
                f"COCO_DIR points to '{coco_dir}', which doesn't exist. "
                "Set COCO_DIR or create the directory."
            )
        
        # Load annotations
        train_captions_file = os.path.join(coco_dir, "annotations/captions_train2017.json")
        val_captions_file = os.path.join(coco_dir, "annotations/captions_val2017.json")
    
        
        # Process annotations to create datasets
        train_dataset = COCOImageDataset._load_from_annotations(train_captions_file, coco_dir, "train")
        
        val_dataset = COCOImageDataset._load_from_annotations(val_captions_file, coco_dir, "val")
        # train_dataset = train_dataset.shuffle(seed=42)
        val_dataset = val_dataset.shuffle(seed=42)
    
        # Limit samples if specified
        if max_train_samples > 0:
            train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
            
        if max_test_samples > 0:
            val_dataset = val_dataset.select(range(min(max_test_samples, len(val_dataset))))
        
        # Apply basic processing WITHOUT loading images
        process_with_prompt = lambda x: COCOImageDataset.process_example_metadata(x, instruction_prompt, think_tokens)
        train_dataset = train_dataset.map(process_with_prompt)
        val_dataset = val_dataset.map(process_with_prompt)
        
        # Set load_image function as format
        # train_dataset.set_transform(COCOImageDataset.load_image_transform)
        # val_dataset.set_transform(COCOImageDataset.load_image_transform)
        
        # Combine into DatasetDict
        return DatasetDict({
            "train": train_dataset,
            "test": val_dataset
        })
    
    @staticmethod
    def _load_from_annotations(annotation_file: str, coco_dir: str, split: str) -> Dataset:
        """
        Load dataset from COCO annotation file
        """
        # Load the JSON
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Extract image info and captions
        images = {img['id']: img for img in data['images']}
        
        records = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            img_info = images[img_id]
            
            # Create a record similar to HuggingFace's COCO dataset structure
            records.append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'image_path': os.path.join(coco_dir, f"{split}2017", img_info['file_name']),
                'sentences': {
                    'raw': ann['caption']
                }
            })
        
        # Convert to DataFrame then to Dataset
        df = pd.DataFrame(records)
        return Dataset.from_pandas(df)
    
    @staticmethod
    def process_example_metadata(example: Dict[str, Any],instruction_prompt: str = "Please write SVG code for generating the image corresponding to the following description:",think_tokens:bool = True) -> Dict[str, Any]:
        """
        Process metadata only (no image loading)
        """
        
        if think_tokens:
            prompt = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
            prompt += f"\nUser: {instruction_prompt} "+ example["sentences"]["raw"] + "\nAssistant: <think>"
        else:
            prompt = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant provides the User with the answer enclosed within <answer> </answer> tags, i.e., <answer> answer here </answer>."
            prompt += f"\nUser: {instruction_prompt} "+ example["sentences"]["raw"] + "\nAssistant: <answer>"
        
        return {
            "prompt": prompt,
            "solution": example["sentences"]["raw"],
            "image_path": example["image_path"],  # Keep path for later loading
            "image_id": example["image_id"]
        }
    
    @staticmethod
    def load_image_transform(examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load image on-demand during dataset access
        """
        images = []
    
        # Process each image path
        for path in examples['image_path']:
            try:
                image = Image.open(path).convert('RGB')
                images.append(image_transform(image))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                images.append(None)
        
        # Add the images to the examples dictionary
        examples['image'] = images
        return examples


class COCOImageDatasetSFT:
    """
    Dataset loader and processor for COCO Image Dataset from local files
    """
    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        instruction_prompt: str = "Please write SVG code for generating the image corresponding to the following description:",
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Load the dataset from local files at ~/data/coco
        """
        coco_dir = os.path.expanduser("~/data/coco")
        
        # Load annotations
        train_captions_file = os.path.join(coco_dir, "annotations/captions_train2017.json")
        val_captions_file = os.path.join(coco_dir, "annotations/captions_val2017.json")
    
        
        # Process annotations to create datasets
        train_dataset = COCOImageDatasetSFT._load_from_annotations(train_captions_file, coco_dir, "train")
        
        val_dataset = COCOImageDatasetSFT._load_from_annotations(val_captions_file, coco_dir, "val")
        # train_dataset = train_dataset.shuffle(seed=42)
        val_dataset = val_dataset.shuffle(seed=42)
    
        # Limit samples if specified
        if max_train_samples > 0:
            train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
            
        if max_test_samples > 0:
            val_dataset = val_dataset.select(range(min(max_test_samples, len(val_dataset))))
        
        # Apply basic processing WITHOUT loading images
        process_with_prompt = lambda x: COCOImageDatasetSFT.process_example_metadata(x, instruction_prompt)
        train_dataset = train_dataset.map(process_with_prompt)
        val_dataset = val_dataset.map(process_with_prompt)
        
        # Set load_image function as format
        # train_dataset.set_transform(COCOImageDataset.load_image_transform)
        # val_dataset.set_transform(COCOImageDataset.load_image_transform)
        
        # Combine into DatasetDict
        return DatasetDict({
            "train": train_dataset,
            "test": val_dataset
        })
    
    @staticmethod
    def _load_from_annotations(annotation_file: str, coco_dir: str, split: str) -> Dataset:
        """
        Load dataset from COCO annotation file
        """
        # Load the JSON
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Extract image info and captions
        images = {img['id']: img for img in data['images']}
        
        records = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            img_info = images[img_id]
            
            # Create a record similar to HuggingFace's COCO dataset structure
            records.append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'image_path': os.path.join(coco_dir, f"{split}2017", img_info['file_name']),
                'sentences': {
                    'raw': ann['caption']
                }
            })
        
        # Convert to DataFrame then to Dataset
        df = pd.DataFrame(records)
        return Dataset.from_pandas(df)
    
    @staticmethod
    def process_example_metadata(example: Dict[str, Any],instruction_prompt: str = "Please write SVG code for generating the image corresponding to the following description:") -> Dict[str, Any]:
        """
        Process metadata only (no image loading)
        """
        return {
            "prompt":  (
            f"User: {instruction_prompt} "
            + example["sentences"]["raw"]
            + "\nAssistant: "
        ),
            "solution": example["sentences"]["raw"],
            "image_path": example["image_path"],  # Keep path for later loading
            "image_id": example["image_id"]
        }
    
    @staticmethod
    def load_image_transform(examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load image on-demand during dataset access
        """
        images = []
    
        # Process each image path
        for path in examples['image_path']:
            try:
                image = Image.open(path).convert('RGB')
                images.append(image_transform(image))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                images.append(None)
        
        # Add the images to the examples dictionary
        examples['image'] = images
        return examples

if __name__ == "__main__":
    # Quick test of dataset loading and image processing
    dataset = COCOImageDataset.load_dataset("coco_image", max_train_samples=0, max_test_samples=1024)
    test_data = dataset["test"]
    
    # Print column names and a sample row
    print("\n=== Dataset Columns and Sample ===")
    
    # Get the column names
    columns = list(test_data.features.keys())
    print(f"Column names: {columns}")
    
    # Print one sample with its column names
    sample_idx = 0  # First sample
    sample = test_data[sample_idx]
    print("\nSample data row:")
    for column in columns:
        if column == "prompt":
            # For long text fields, just show first 80 characters
            value = f"{sample[column][:80]}..." if len(sample[column]) > 80 else sample[column]
        else:
            value = sample[column]
        print(f"  {column}: {value}")
        
    