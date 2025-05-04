from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, IterableDataset, DatasetDict
import os
import torch
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think>\n<answer> answer here </answer>"
)

class VGBenchDataset:
    """
    Dataset loader and processor for XDG Dataset
    """
    

    
    @staticmethod
    def load_dataset(
        dataset_name: str = "vgbench/VGen",
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        use_local_cache: bool = True,
        local_cache_path: str = "~/.cache/huggingface/hub/datasets--vgbench--VGen/snapshots/6133b6d8fba3602781d3cc4b46142a2e44376e01/svg.json",
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Load the dataset from HuggingFace or local source
        """
        if use_local_cache:
        # Expand user directory if needed (for ~)
            local_path = os.path.expanduser(local_cache_path)
            
            print(f"Loading dataset from local cache: {local_path}")
            
            # Load JSON data from local file
            try:
                dataset = load_dataset('json', data_files=local_path)
                
                # The JSON loading creates a 'train' split, but we want to match
                # the original dataset structure which might have validation/test
                splits = {}
                
                # Check the loaded dataset structure
                if 'train' in dataset:
                    # Rename 'train' to match expected structure
                    splits['train'] = dataset['train']
                    
                    
                
                dataset = DatasetDict(splits)
                print(f"Loaded dataset with splits: {list(dataset.keys())}")
            except Exception as e:
                print(f"Error loading from local cache: {e}")
                print("Falling back to HuggingFace Hub...")
                dataset = load_dataset(dataset_name)
        else:
            # Use standard HuggingFace loading
            dataset = load_dataset(dataset_name)
    
        
        for split in dataset:
            if "solution"==dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("solution")
        
        
        dataset[split] = dataset[split].rename_column("code", "svg")
        dataset[split] = dataset[split].rename_column("caption", "solution")
        
        dataset = dataset.map(VGBenchDataset.process_example)
        
        
        dataset = dataset.filter(
            lambda example: example['vformat']=='svg'
        )
        
       
        if max_train_samples and max_train_samples > 0:
            for split in dataset:
                if split == "train":
                    dataset[split] = dataset[split].select(range(min(max_train_samples, len(dataset[split]))))
        
        if max_test_samples and max_test_samples > 0:       
            for split in dataset:
                if split == "validation":
                    dataset[split] = dataset[split].select(range(min(max_test_samples, len(dataset[split]))))    
        return dataset
    
    @staticmethod
    def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
       
        # Ensure the svg field exists
        svg = example.get('svg', '')
        description = example.get('solution', '')
        instruction_prompt = "Please write SVG code for generating the image corresponding to the following description:"
        # Create the formatted prompt
        prompt = (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. " 
            +f"The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {instruction_prompt} "
            + description
            + "\nAssistant: <think>"
        )
        
        # Return all fields from the original example, plus the formatted prompt and solution
        return {
            **example,
            "prompt": prompt,
            # Keep svg field as is
        }
            
