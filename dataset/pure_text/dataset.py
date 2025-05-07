from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, IterableDataset
import torch
import pandas as pd
from datasets import DatasetDict
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think>\n<answer> answer here </answer>"
)
class PureTextDataset:
    """
    Dataset loader and processor for text-based shape descriptions
    """
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = 100000,
        max_test_samples: Optional[int] = -1,
        mixture_coeff: Optional[Dict[str, float]] = {
                'shapes': 1.0,
                'quantity': 1.0,
                'relation': 1.0
            },
        random_seed: int = 42,  # Added random seed parameter
        data_type = "daily",
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Create a dataset by reading and mixing JSON files of simple shapes and compositions.
        
        Args:
            dataset_name: Name of the dataset (unused)
            dataset_config: Configuration for the dataset (unused)
            max_train_samples: Maximum number of training samples (-1 for all)
            max_test_samples: Maximum number of test samples (-1 for all)
            mixture_coeff: Dictionary with mixing coefficients for each source file
                          Keys: 'shapes', 'quantity', 'relation'
                          Values: Relative weights for mixing (default: equal weights)
            random_seed: Seed for random operations to ensure reproducibility (default: 42)
        
        Returns:
            DatasetDict with 'train' and 'validation' splits
        """
        import os
        import json
        import random
        from datasets import Dataset, DatasetDict
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
        
        # Default mixture coefficients (equal weighting)
        
        # Normalize coefficients to sum to 1
        total = sum(mixture_coeff.values())
        mixture_coeff = {k: v / total for k, v in mixture_coeff.items()}
        
        # Define paths to JSON files
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if data_type == "simple":
            shapes_path = os.path.join(base_dir, 'simple_shapes.json')
            quantity_path = os.path.join(base_dir, 'simple_compositions_quantity.json')
            relation_path = os.path.join(base_dir, 'simple_compositions_relation.json')
        elif data_type == "daily":
            shapes_path = os.path.join(base_dir, 'daily_shapes.json')
            quantity_path = os.path.join(base_dir, 'daily_compositions_quantity.json')
            relation_path = os.path.join(base_dir, 'daily_compositions_relation.json')
        # Load data from JSON files
        shapes_data = []
        quantity_data = []
        relation_data = []
        
        try:
            with open(shapes_path, 'r') as f:
                shapes_data = json.load(f)
                print(f"Loaded {len(shapes_data)} shape descriptions")
        except FileNotFoundError:
            print(f"Warning: {shapes_path} not found")
        
        try:
            with open(quantity_path, 'r') as f:
                quantity_data = json.load(f)
                print(f"Loaded {len(quantity_data)} quantity descriptions")
        except FileNotFoundError:
            print(f"Warning: {quantity_path} not found")
        
        try:
            with open(relation_path, 'r') as f:
                relation_data = json.load(f)
                print(f"Loaded {len(relation_data)} relation descriptions")
        except FileNotFoundError:
            print(f"Warning: {relation_path} not found")
        
        # Shuffle all data first (using consistent seed)
        random.shuffle(shapes_data)
        random.shuffle(quantity_data)
        random.shuffle(relation_data)
        
        # Create validation set by holding out 250 samples each from quantity and relation data
        val_quantity = []
        val_relation = []
        
        if len(quantity_data) >= 250:
            val_quantity = quantity_data[:250]
            quantity_data = quantity_data[250:]  # Remove validation samples from training pool
        else:
            val_quantity = quantity_data.copy() if quantity_data else []
            quantity_data = []
            
        if len(relation_data) >= 250:
            val_relation = relation_data[:250]
            relation_data = relation_data[250:]  # Remove validation samples from training pool
        else:
            val_relation = relation_data.copy() if relation_data else []
            relation_data = []
            
        # Create validation dataset
        all_val_samples = val_quantity + val_relation
        random.shuffle(all_val_samples)
        print(f"Created validation set with {len(all_val_samples)} samples ({len(val_quantity)} quantity, {len(val_relation)} relation)")
        
        # Determine dataset sizes for training based on mixture coefficients
        total_train_samples = max_train_samples if max_train_samples > 0 else 100000  # Default to 100k if no limit
        
        shapes_count = int(total_train_samples * mixture_coeff['shapes'])
        quantity_count = int(total_train_samples * mixture_coeff['quantity'])
        relation_count = total_train_samples - shapes_count - quantity_count  # Ensure we get exactly the requested count
        
        print(f"Training mixture proportions - shapes: {shapes_count}, quantity: {quantity_count}, relation: {relation_count}")
        
        # Sample from each source according to the mixture coefficients (with replacement if needed)
        shapes_samples = random.choices(shapes_data, k=shapes_count) if shapes_data else []
        quantity_samples = random.choices(quantity_data, k=quantity_count) if quantity_data else []
        relation_samples = random.choices(relation_data, k=relation_count) if relation_data else []
        
        # Combine all training samples
        all_train_samples = shapes_samples + quantity_samples + relation_samples
        random.shuffle(all_train_samples)  # Shuffle to mix the sources
        
        print(f"Final training dataset: {len(all_train_samples)} total samples")
        print(f"  - {len(shapes_samples)} shapes")
        print(f"  - {len(quantity_samples)} quantity")
        print(f"  - {len(relation_samples)} relation")
        
        # Create the dataset objects
        train_data = []
        instruction_prompt = "Please write SVG code for generating the image corresponding to the following description:"
        for description in all_train_samples:
            train_data.append({
                "prompt": f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {instruction_prompt} {description['caption']}\nAssistant: <think>",
                "solution": description['caption'],
            })
        
        val_data = []
        for description in all_val_samples:
            val_data.append({
                 "prompt": f"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {instruction_prompt} {description['caption']}\nAssistant: <think>",
                "solution": description['caption'],
            })
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        validation_dataset = Dataset.from_list(val_data)
        
        # Apply max_test_samples limit if specified
        if max_test_samples > 0 and max_test_samples < len(validation_dataset):
            validation_dataset = validation_dataset.select(range(max_test_samples))
            print(f"Limiting validation set to {max_test_samples} samples")
        
        # Create dataset dictionary
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": validation_dataset
        })
        
        return dataset_dict
    

if __name__ == "__main__":
    def test_dataset():
        """Simple test function to demonstrate dataset loading."""
        print("\n=== Testing PureTextDataset ===")
        
        # Test with default settings
        print("\nTest 1: Default settings with limited samples")
        dataset1 = PureTextDataset.load_dataset(
            dataset_name="pure_text",
            max_train_samples=100,
            max_test_samples=20,
            random_seed=42
        )
        
        print(f"Dataset splits: {list(dataset1.keys())}")
        print(f"Training samples: {len(dataset1['train'])}")
        print(f"Validation samples: {len(dataset1['test'])}")
        
        # Show example
        print("\nExample training sample:")
        example = dataset1['train'][0]
        print(f"User prompt: {example['prompt']}")
        print(f"Solution: {example['solution']}")
        
        # Test with custom mixture coefficients
        print("\nTest 2: Custom mixture coefficients")
        dataset2 = PureTextDataset.load_dataset(
            dataset_name="pure_text",
            max_train_samples=200,
            mixture_coeff={
                'shapes': 0.5,    # 50% shapes
                'quantity': 0.3,  # 30% quantity
                'relation': 0.2,  # 20% relation
            },
            random_seed=123
        )
        
        print(f"Training samples: {len(dataset2['train'])}")
        
        # Test data consistency with same seed
        print("\nTest 3: Checking reproducibility with same seed")
        dataset3a = PureTextDataset.load_dataset(
            dataset_name="pure_text",
            max_train_samples=50,
            random_seed=456
        )
        
        dataset3b = PureTextDataset.load_dataset(
            dataset_name="pure_text", 
            max_train_samples=50,
            random_seed=456
        )
        
        # Check if first 3 examples are the same
        print("First 3 examples from two datasets with same seed:")
        are_same = all(
            dataset3a['train'][i]['solution'] == dataset3b['train'][i]['solution'] 
            for i in range(min(3, len(dataset3a['train'])))
        )
        print(f"Datasets are identical: {are_same}")

    # Run the test
    test_dataset()