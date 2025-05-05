from typing import Dict, List, Optional, Union, Any
from datasets import Dataset, IterableDataset
import json
import os
import torch

class NewHQSVGDataset:
    """
    Dataset loader and processor for SVG data from structured JSON files
    """
    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        train_path: str = "/home/chenyamei/data/haoquan_svg/train.json",
        test_path: str = "/home/chenyamei/data/haoquan_svg/eval.json",
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        filter_successful_only: bool = False,
        filter_has_description: bool = True,
        filter_has_svg: bool = True,
        filter_text_content: bool = False, 
        instruction_prompt: str = "Please write SVG code for generating the image corresponding to the following description:",
        seed: int = 42,  # For reproducibility
        **kwargs
    ) -> Dict[str, Dataset]:
        """
        Load the dataset from JSON files
        
        Args:
            train_path: Path to the training JSON file
            test_path: Path to the test JSON file
            max_train_samples: Maximum number of training samples (-1 for all)
            max_test_samples: Maximum number of test samples (-1 for all)
            filter_successful_only: Whether to filter only successful SVG generations
            filter_has_description: Whether to filter examples with non-empty descriptions
            filter_text_content: Whether to filter out text-related content
            instruction_prompt: Prompt to prepend to the description
            seed: Random seed for shuffling
            **kwargs: Additional arguments
        """
        train_data = []
        test_data = []
        
        # Load training data
        if os.path.exists(train_path):
            with open(train_path, 'r') as f:
                train_data = json.load(f)
            print(f"Loaded {len(train_data)} examples from {train_path}")
        else:
            print(f"Warning: Training file not found: {train_path}")
        
        # Load test data
        if os.path.exists(test_path):
            with open(test_path, 'r') as f:
                test_data = json.load(f)
            print(f"Loaded {len(test_data)} examples from {test_path}")
        else:
            print(f"Warning: Test file not found: {test_path}")
        
        # Filter training data
        filtered_train = NewHQSVGDataset._filter_data(
            train_data,
            filter_successful_only,
            filter_has_description,
            filter_has_svg,
            filter_text_content
        )
        
        # Filter test data
        filtered_test = NewHQSVGDataset._filter_data(
            test_data,
            filter_successful_only,
            filter_has_description,
            filter_has_svg,
            filter_text_content
        )
        
        print(f"After filtering: {len(filtered_train)} train examples, {len(filtered_test)} test examples")
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_list(filtered_train)
        test_dataset = Dataset.from_list(filtered_test)
        
        # Shuffle datasets
        train_dataset = train_dataset.shuffle(seed=seed)
        
        # Process examples
        process_with_instruction = lambda example: NewHQSVGDataset.process_example(
            example, instruction_prompt=instruction_prompt
        )
        train_dataset = train_dataset.map(process_with_instruction)
        test_dataset = test_dataset.map(process_with_instruction)
        
        # Apply selection if needed
        if max_train_samples and max_train_samples > 0:
            train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
        
        if max_test_samples and max_test_samples > 0:       
            test_dataset = test_dataset.select(range(min(max_test_samples, len(test_dataset))))
        
        return {
            'train': train_dataset,
            'test': test_dataset
        }
    
    @staticmethod
    def _filter_data(data, filter_successful_only, filter_has_description,filter_has_svg, filter_text_content):
        """Helper method to filter data according to criteria"""
        filtered_data = data
        
        # Filter by API call success
        if filter_successful_only:
            filtered_data = [
                item for item in filtered_data 
                if item.get('api_call_success', False) 
                and item.get('classification', {}).get('is_of_class', False)
            ]
            print(f"After success filtering: {len(filtered_data)} examples")
        
        # Filter by description
        if filter_has_description:
            filtered_data = [
                item for item in filtered_data 
                if item.get('description', '').strip()
            ]
            print(f"After description filtering: {len(filtered_data)} examples")
        # Filter by SVG content
        if filter_has_svg:  
            filtered_data = [
                item for item in filtered_data 
                if item.get('svg_content', '').strip()
            ]
            print(f"After nonempty SVG content filtering: {len(filtered_data)} examples")
        
        # Filter text-related content
        if filter_text_content:
            # Define terms to filter out
            text_terms = [' text', ' letter', ' character', '"', "'", "word","symbol", "Symbol"]
            
            # Filter function that excludes examples containing any of these terms
            filtered_data = [
                item for item in filtered_data 
                if not any(term.lower() in item.get('description', '').lower() for term in text_terms)
            ]
            print(f"After text content filtering: {len(filtered_data)} examples")
        
        return filtered_data
    
    @staticmethod
    def process_example(example: Dict[str, Any], instruction_prompt: str = "Please write SVG code for generating the image corresponding to the following description:") -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
        # Get SVG content and description
        svg = example.get('svg_content', '')
        description = example.get('description', '')
        
        # Create the formatted prompt
        prompt = (
            "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. " 
            +f"The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {instruction_prompt} "
            + description
            + "\nAssistant: <think>"
        )
        
        # Return formatted example
        return {
            **example,
            "prompt": prompt,
            "solution": description,
            "svg": svg,  # Rename svg_content to svg for consistency
        }
    
    @staticmethod
    def create_metadata_fields(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper method to create metadata fields for visualization and analysis
        """
        metadata = {
            "classification_info": {
                "major_class": example.get("major_class", ""),
                "sub_class": example.get("sub_class", ""),
                "appearance": example.get("appearance", ""),
                "classification": example.get("classification", {}),
            },
            "success": example.get("api_call_success", False),
            "file_name": example.get("file_name", "")
        }
        
        return metadata

# Add this to the if __name__ == "__main__" block to test the new dataset loader
if __name__ == "__main__":
    # Test the new dataset loader
    dataset = NewHQSVGDataset.load_dataset(
        None,None,
        max_train_samples=-1,
        max_test_samples=-1,
        filter_successful_only=True,
    )

    # Print dataset statistics
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    
    # Show a sample entry
    print("\nSample training example:")
    sample = dataset['train'][0]
    print(f"Description: {sample['description']}")
    print(f"Prompt: {sample['prompt'][:100]}...")
    print(f"SVG length: {len(sample['svg'])}")