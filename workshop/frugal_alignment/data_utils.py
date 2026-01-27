"""
Data utilities for preparing datasets for Llama-3 fine-tuning
"""

from datasets import load_dataset, Dataset
import json
from pathlib import Path
from typing import List, Dict, Optional


def load_json_dataset(file_path: str) -> Dataset:
    """
    Load dataset from JSON file.
    
    Expected format:
    [
        {"instruction": "...", "response": "..."},
        {"instruction": "...", "response": "..."},
        ...
    ]
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        HuggingFace Dataset
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return Dataset.from_list(data)


def format_chat_template(instruction: str, response: str, system_prompt: Optional[str] = None) -> str:
    """
    Format instruction and response into a chat template.
    
    Args:
        instruction: User instruction/question
        response: Model response
        system_prompt: Optional system prompt
        
    Returns:
        Formatted prompt string
    """
    if system_prompt:
        prompt = f"""### System:
{system_prompt}

### Instruction:
{instruction}

### Response:
{response}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{response}"""
    
    return prompt


def format_alpaca_style(examples: Dict) -> Dict[str, List[str]]:
    """
    Format examples in Alpaca style.
    
    Args:
        examples: Batch of examples with 'instruction' and 'response' fields
        
    Returns:
        Dictionary with formatted 'text' field
    """
    texts = []
    for instruction, response in zip(examples['instruction'], examples['response']):
        text = format_chat_template(instruction, response)
        texts.append(text)
    
    return {"text": texts}


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    Create a sample dataset for testing.
    
    Args:
        output_path: Path to save the dataset
        num_samples: Number of samples to generate
    """
    samples = [
        {
            "instruction": f"What is {i} + {i+1}?",
            "response": f"The sum of {i} and {i+1} is {2*i+1}."
        }
        for i in range(num_samples)
    ]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Created sample dataset with {num_samples} examples at {output_path}")


def prepare_dataset_for_training(
    dataset_name_or_path: str,
    is_local: bool = False,
    split: str = "train"
) -> Dataset:
    """
    Prepare dataset for training.
    
    Args:
        dataset_name_or_path: HuggingFace dataset name or local file path
        is_local: Whether the dataset is local
        split: Dataset split to use
        
    Returns:
        Prepared dataset
    """
    if is_local:
        dataset = load_json_dataset(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path, split=split)
    
    # Apply formatting
    dataset = dataset.map(format_alpaca_style, batched=True)
    
    return dataset


if __name__ == "__main__":
    # Example: Create a sample dataset
    create_sample_dataset("./data/sample_dataset.json", num_samples=50)
