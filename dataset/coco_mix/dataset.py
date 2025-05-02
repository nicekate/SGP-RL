from typing import Dict, List, Optional, Union, Any
import os
import json
from datasets import Dataset, IterableDataset, DatasetDict
import torch
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])



from typing import Dict, List, Optional, Union, Any
from datasets import Dataset, IterableDataset, DatasetDict, concatenate_datasets
import random
import os
import sys

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset.haoquan_svg.dataset import HQSVGDataset
from dataset.coco_image.dataset import COCOImageDataset


class MixedSVGImageDataset:
    """
    Dataset loader that mixes COCO images and HaoQuan SVG datasets in a 1:1 ratio
    by first determining available data in both datasets.
    """
    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        svg_dataset_path: str = "/home/chenyamei/data/haoquan_svg/svg-gen-70k.jsonl",
        filter_successful_only: bool = True,
        filter_has_description: bool = True,
        complexity_threshold: Optional[float] = None,
        instruction_prompt: str = "Please write SVG code for generating the image corresponding to the following description:",
        seed: int = 42,
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Load a mixed dataset with equal parts COCO images and HaoQuan SVG.
        First loads ALL COCO data, then selects examples to match SVG dataset.
        
        Args:
            dataset_name: Name identifier for the dataset
            dataset_config: Configuration dictionary
            max_train_samples: Maximum number of final training samples after mixing
            max_test_samples: Maximum number of final test samples after mixing
            svg_dataset_path: Path to the SVG JSONL file
            filter_successful_only: Whether to filter only successful SVG generations
            filter_has_description: Whether to filter examples with non-empty descriptions
            complexity_threshold: Filter examples with overall_complexity above this threshold
            seed: Random seed for shuffling and splitting
            **kwargs: Additional arguments
        """
        print(f"Loading mixed dataset with data from HaoQuan SVG and COCO images...")
        
        # Step 1: First load the full COCO dataset
        print("Loading full COCO dataset...")
        coco_dataset = COCOImageDataset.load_dataset(
            dataset_name="coco_image",
            max_train_samples=-1,  # Load all available COCO data
            max_test_samples=-1,   # Load all available COCO data
            instruction_prompt = instruction_prompt,
        )
        
        # Get sizes of original COCO dataset
        coco_train_size_original = len(coco_dataset['train'])
        coco_test_size_original = len(coco_dataset['test'])
        print(f"Loaded {coco_train_size_original} training and {coco_test_size_original} testing examples from COCO dataset")
        
        # Step 2: Shuffle COCO datasets
        coco_dataset['train'] = coco_dataset['train'].shuffle(seed=seed)
        # coco_dataset['test'] = coco_dataset['test'].shuffle(seed=seed)
        
        # Step 3: Now load SVG dataset
        print("Loading SVG dataset...")
        svg_dataset = HQSVGDataset.load_dataset(
            dataset_name="haoquan_svg",
            dataset_path=svg_dataset_path,
            max_train_samples=-1,
            max_test_samples=-1,
            filter_successful_only=filter_successful_only,
            filter_has_description=filter_has_description,
            complexity_threshold=complexity_threshold,
            instruction_prompt = instruction_prompt,
            seed=seed,
        )
        
        # Get sizes of SVG dataset splits
        svg_train_size = len(svg_dataset['train'])
        svg_test_size = len(svg_dataset['test'])
        print(f"Loaded {svg_train_size} training and {svg_test_size} testing examples from SVG dataset")
        
        # Step 4: Select balanced number of examples
        # If one dataset is smaller, use that to determine the sample count per source
        train_samples_per_source = min(svg_train_size, coco_train_size_original)
        test_samples_per_source = min(svg_test_size, coco_test_size_original)
        
        print(f"Using {train_samples_per_source} training and {test_samples_per_source} testing samples from each dataset")
        
        # Select the appropriate number of examples from COCO
        coco_dataset['train'] = coco_dataset['train'].select(range(train_samples_per_source))
        coco_dataset['test'] = coco_dataset['test'].select(range(test_samples_per_source))
        
        # Select the appropriate number of examples from SVG if needed
        if svg_train_size > train_samples_per_source:
            svg_dataset['train'] = svg_dataset['train'].select(range(train_samples_per_source))
        if svg_test_size > test_samples_per_source:
            svg_dataset['test'] = svg_dataset['test'].select(range(test_samples_per_source))
        
        # Step 5: Add cross-dataset fields
        # Add 'svg' field as None to COCO dataset
        def add_svg_field(example):
            example['svg'] = None
            example['dataset_source'] = 'coco'
            return example
        
        coco_dataset['train'] = coco_dataset['train'].map(add_svg_field)
        coco_dataset['test'] = coco_dataset['test'].map(add_svg_field)
        
        # Add 'image_path' field as None to SVG dataset
        def add_image_path_field(example):
            example['image_path'] = None
            example['dataset_source'] = 'svg'
            return example
        
        svg_dataset['train'] = svg_dataset['train'].map(add_image_path_field)
        svg_dataset['test'] = svg_dataset['test'].map(add_image_path_field)
        
        # Step 6: Ensure datasets have the same schema
        columns = ['prompt', 'solution', 'svg', 'image_path', 'dataset_source']
        
        # Process COCO dataset columns
        coco_columns = coco_dataset['train'].column_names
        coco_to_remove = [col for col in coco_columns if col not in columns]
        if coco_to_remove:
            coco_dataset['train'] = coco_dataset['train'].remove_columns(coco_to_remove)
            coco_dataset['test'] = coco_dataset['test'].remove_columns(coco_to_remove)
        
        # Process SVG dataset columns
        svg_columns = svg_dataset['train'].column_names
        svg_to_remove = [col for col in svg_columns if col not in columns]
        if svg_to_remove:
            svg_dataset['train'] = svg_dataset['train'].remove_columns(svg_to_remove)
            svg_dataset['test'] = svg_dataset['test'].remove_columns(svg_to_remove)
        
        # Step 7: Concatenate datasets and shuffle
        mixed_train = concatenate_datasets([coco_dataset['train'], svg_dataset['train']])
        mixed_test = concatenate_datasets([coco_dataset['test'], svg_dataset['test']])
        
        mixed_train = mixed_train.shuffle(seed=seed)
        mixed_test = mixed_test.shuffle(seed=seed)
        
        # Step 8: Apply max_train_samples and max_test_samples limits if specified
        if max_train_samples > 0 and max_train_samples < len(mixed_train):
            print(f"Limiting combined training data from {len(mixed_train)} to {max_train_samples} samples")
            mixed_train = mixed_train.select(range(max_train_samples))
        
        if max_test_samples > 0 and max_test_samples < len(mixed_test):
            print(f"Limiting combined test data from {len(mixed_test)} to {max_test_samples} samples")
            mixed_test = mixed_test.select(range(max_test_samples))
        
        # Create final dataset dictionary
        mixed_dataset = DatasetDict({
            'train': mixed_train,
            'test': mixed_test
        })
        
        # Count dataset sources after all filtering
        train_coco_count = sum(1 for x in mixed_train if x['dataset_source'] == 'coco')
        train_svg_count = sum(1 for x in mixed_train if x['dataset_source'] == 'svg')
        test_coco_count = sum(1 for x in mixed_test if x['dataset_source'] == 'coco')
        test_svg_count = sum(1 for x in mixed_test if x['dataset_source'] == 'svg')
        
        print(f"Final mixed dataset created:")
        print(f"  Train: {len(mixed_dataset['train'])} examples "
            f"({train_coco_count} COCO [{train_coco_count/len(mixed_train):.1%}], "
            f"{train_svg_count} SVG [{train_svg_count/len(mixed_train):.1%}])")
        print(f"  Test: {len(mixed_dataset['test'])} examples "
            f"({test_coco_count} COCO [{test_coco_count/len(mixed_test):.1%}], "
            f"{test_svg_count} SVG [{test_svg_count/len(mixed_test):.1%}])")
        
        return mixed_dataset

if __name__ == "__main__":
    # Sample usage
    dataset = MixedSVGImageDataset.load_dataset(
        'mixed_svg_image',
        max_train_samples=-1,  # 500 from each source
        max_test_samples=500,    # 100 from each source
        filter_successful_only=True,
    )
    
    # Print sample entries
    print("\nSample COCO entry:")
    coco_sample = next(x for x in dataset['train'] if x['dataset_source'] == 'coco')
    for k, v in coco_sample.items():
        if k != 'prompt':  # Skip printing full prompt
            print(f"{k}: {v}")
    
    print("\nSample SVG entry:")
    svg_sample = next(x for x in dataset['train'] if x['dataset_source'] == 'svg')
    for k, v in svg_sample.items():
        if k != 'prompt' and k != 'svg':  # Skip printing full SVG and prompt
            print(f"{k}: {v}")