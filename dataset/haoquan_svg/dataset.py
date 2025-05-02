from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, IterableDataset
import json
import os
import torch



class HQSVGDataset:
    """
    Dataset loader and processor for SVG data in JSONL format
    """
    
    @staticmethod
    def load_dataset(
        dataset_name: str,
        dataset_config: Optional[Dict[str, Any]] = None,
        dataset_path: str="/home/chenyamei/data/haoquan_svg/svg-gen-70k.jsonl",
        test_split_ratio: float = 0.0,  
        max_train_samples: Optional[int] = -1,
        max_test_samples: Optional[int] = -1,
        filter_successful_only: bool = True,
        filter_has_description: bool = True,
        filter_text_content: bool = True, 
        complexity_threshold: Optional[float] = None,
        instruction_prompt: str = "Please write SVG code for generating the image corresponding to the following description:",
        seed: int = 42,  # Added seed parameter for reproducibility
        **kwargs
    ) -> Union[Dataset, IterableDataset]:
        """
        Load the dataset from a JSONL file
        
        Args:
            dataset_path: Path to the JSONL file or directory containing the file
            test_split_ratio: Ratio of examples to use for testing (default: 0.2 for 80/20 split)
            max_train_samples: Maximum number of training samples (-1 for all)
            max_test_samples: Maximum number of test samples (-1 for all)
            filter_successful_only: Whether to filter only successful SVG generations
            filter_has_description: Whether to filter examples with non-empty descriptions
            complexity_threshold: Filter examples with overall_complexity above this threshold
            seed: Random seed for shuffling and splitting
            **kwargs: Additional arguments
        """
        # Load from JSONL file
        if os.path.isdir(dataset_path):
            # If a directory, look for jsonl files
            dataset = load_dataset('json', data_files={
                'train': os.path.join(dataset_path, '*.jsonl')
            })
        else:
            # Direct file path
            dataset = load_dataset('json', data_files={
                'train': dataset_path
            })
        
        # Apply filters
        if filter_successful_only:
            dataset['train'] = dataset['train'].filter(
                lambda example: example.get('success', False)
            )
        if filter_has_description:
            dataset['train'] = dataset['train'].filter(
                lambda example: example.get('description', '') 
            )
         # Filter out text-related content
        if filter_text_content:
            # Define terms to filter out
            text_terms = ['text', 'letter', 'character', '"', "sorry", "symbol", "Symbol"]
            
            # Filter function that excludes examples containing any of these terms
            def filter_text_related(example):
                description = example.get('description', '').lower()
                return not any(term in description.lower() for term in text_terms)
            
            # Apply filter
            dataset['train'] = dataset['train'].filter(filter_text_related)
            print(f"After filtering out text-related content: {len(dataset['train'])} examples")
        
        if complexity_threshold is not None:
            dataset['train'] = dataset['train'].filter(
                lambda example: example.get('overall_complexity', 0) > complexity_threshold
            )
        
        # Shuffle the dataset before splitting
        dataset['train'] = dataset['train'].shuffle(seed=seed)
        
        # Split into train and test (80/20 split by default)
        if test_split_ratio > 0:
            dataset = dataset['train'].train_test_split(test_size=test_split_ratio, seed=seed)
        else:
            dataset['test'] = dataset['train'].select(range(10))
        # Process examples to match expected format
        process_with_instruction = lambda example: HQSVGDataset.process_example(
        example, instruction_prompt=instruction_prompt
            )
        dataset = dataset.map(process_with_instruction)
        
        # Apply selection if needed
        if max_train_samples and max_train_samples > 0:
            dataset['train'] = dataset['train'].select(range(min(max_train_samples, len(dataset['train']))))
        
        if max_test_samples and max_test_samples > 0:       
            dataset['test'] = dataset['test'].select(range(min(max_test_samples, len(dataset['test']))))
            
        return dataset
    @staticmethod
    def process_example(example: Dict[str, Any], instruction_prompt: str = "Please write SVG code for generating the image corresponding to the following description:") -> Dict[str, Any]:
        """
        Process a single example from the dataset
        """
        # Ensure the svg field exists
        svg = example.get('svg', '')
        description = example.get('description', '')
        
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
            "solution": description,
            # Keep svg field as is
        }

    @staticmethod
    def create_metadata_fields(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper method to create metadata fields for visualization and analysis
        """
        metadata = {
            "complexity_info": {
                "description_length": example.get("description_length", 0),
                "element_count": example.get("element_count", 0),
                "element_complexity": example.get("element_complexity", 0),
                "structural_complexity": example.get("structural_complexity", 0),
                "overall_complexity": example.get("overall_complexity", 0),
            },
            "success": example.get("success", False),
            "id": example.get("id", "")
        }
        
        return metadata
    
if __name__ == "__main__":
    # Sample usage

    # Load the dataset
    dataset = HQSVGDataset.load_dataset(
        'haoquan_svg',
        dataset_path="/home/chenyamei/data/haoquan_svg/svg-gen-70k.jsonl",
        test_split_ratio=0.0,
        max_train_samples=-1,
        filter_successful_only=True,
        complexity_threshold=0.0  # Only examples with complexity > 5
    )

    # Print dataset statistics
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    print(dataset['train'][0])
    # Show sample entry
    # print(dataset['train'].filter(lambda x: 'Gray distorted and mirrored text creates a cryptic eye chart illusion.' in x['description'])[0]['svg'])