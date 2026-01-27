"""
Llama-3-8B Fine-tuning Script using Unsloth
Efficient training with LoRA/QLoRA for alignment tasks
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import wandb
import yaml
import argparse
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config):
    """
    Load and prepare Llama-3-8B model with Unsloth optimizations.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        model, tokenizer: Prepared model and tokenizer
    """
    model_config = config['model']
    lora_config = config['lora']
    
    # Load model with 4-bit quantization for memory efficiency
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['name'],
        max_seq_length=model_config['max_seq_length'],
        dtype=None,  # Auto-detect optimal dtype
        load_in_4bit=model_config['load_in_4bit'],
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],  # LoRA rank
        target_modules=lora_config['target_modules'],
        lora_alpha=lora_config['alpha'],
        lora_dropout=lora_config['dropout'],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=config['training']['seed'],
    )
    
    return model, tokenizer


def prepare_dataset(config, tokenizer):
    """
    Load and prepare dataset for training.
    
    Args:
        config: Configuration dictionary
        tokenizer: Model tokenizer
        
    Returns:
        Dataset ready for training
    """
    dataset_config = config['dataset']
    
    # Load dataset
    if dataset_config['type'] == 'huggingface':
        dataset = load_dataset(
            dataset_config['name'],
            split=dataset_config['split']
        )
    elif dataset_config['type'] == 'local':
        dataset = load_dataset(
            'json',
            data_files=dataset_config['path'],
            split='train'
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config['type']}")
    
    # Format prompts
    def format_prompts(examples):
        """Format examples into chat template."""
        texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            text = f"""### Instruction:
{instruction}

### Response:
{response}"""
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(format_prompts, batched=True)
    
    return dataset


def train(config):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    """
    # Initialize Weights & Biases
    if config['training']['use_wandb']:
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['run_name'],
            config=config
        )
    
    # Setup model and tokenizer
    print("Loading model...")
    model, tokenizer = setup_model(config)
    
    # Prepare dataset
    print("Loading dataset...")
    dataset = prepare_dataset(config, tokenizer)
    
    # Training arguments
    training_config = config['training']
    output_dir = Path(training_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        per_device_train_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        warmup_steps=training_config['warmup_steps'],
        num_train_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=training_config['logging_steps'],
        optim=training_config['optimizer'],
        weight_decay=training_config['weight_decay'],
        lr_scheduler_type=training_config['lr_scheduler'],
        seed=training_config['seed'],
        output_dir=str(output_dir),
        save_strategy="steps",
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        report_to="wandb" if training_config['use_wandb'] else "none",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        args=training_args,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    final_output_dir = output_dir / "final_model"
    model.save_pretrained(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    # Save merged model (optional - for inference)
    if config['training']['save_merged_model']:
        print("Saving merged 16-bit model...")
        merged_output_dir = output_dir / "merged_model"
        model.save_pretrained_merged(
            str(merged_output_dir),
            tokenizer,
            save_method="merged_16bit",
        )
    
    print("Training complete!")
    
    if config['training']['use_wandb']:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3-8B with Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
