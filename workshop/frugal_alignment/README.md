# Llama-3-8B Fine-tuning with Unsloth

Efficient fine-tuning of Llama-3-8B using Unsloth library with LoRA/QLoRA for alignment tasks.

## Features

- ✅ **Efficient Training**: 4-bit quantization with QLoRA
- ✅ **Fast**: Unsloth optimizations for 2x faster training
- ✅ **Memory Efficient**: Train on consumer GPUs with gradient checkpointing
- ✅ **Flexible**: Easy configuration via YAML
- ✅ **Tracking**: Weights & Biases integration

## Requirements

> **⚠️ Important**: Unsloth requires **Python 3.10 or higher**. It will NOT work with Python 3.8 or 3.9.

### Hardware
- GPU with at least 16GB VRAM (e.g., RTX 3090, RTX 4090, A100)
- CUDA 11.8+ or higher

### Software
- Python 3.10+
- CUDA toolkit

## Installation

### 1. Create Python 3.10+ Virtual Environment

```bash
# Using Python 3.10 or higher
python3.10 -m venv venv-llama
source venv-llama/bin/activate  # On macOS/Linux
# OR
venv-llama\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install transformers datasets peft trl bitsandbytes accelerate wandb pyyaml

# Install Unsloth (requires git)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Quick Start

### 1. Configure Training

Edit `config.yaml` to customize:
- Model parameters
- LoRA settings
- Dataset source
- Training hyperparameters

### 2. Prepare Dataset

**Option A: Use HuggingFace Dataset**
```yaml
dataset:
  type: "huggingface"
  name: "yahma/alpaca-cleaned"
  split: "train"
```

**Option B: Use Local JSON Dataset**
```yaml
dataset:
  type: "local"
  path: "./data/training_data.json"
```

Format your JSON data as:
```json
[
  {
    "instruction": "Your instruction here",
    "response": "Expected response here"
  },
  ...
]
```

You can create a sample dataset using:
```bash
python data_utils.py
```

### 3. Login to Weights & Biases (Optional)

```bash
wandb login
```

To disable WandB, set `use_wandb: false` in `config.yaml`.

### 4. Start Training

```bash
python train.py --config config.yaml
```

## Training Configuration

Key parameters in `config.yaml`:

### Model Settings
- `max_seq_length`: Maximum sequence length (default: 2048)
- `load_in_4bit`: Enable 4-bit quantization (recommended)

### LoRA Settings
- `r`: LoRA rank (default: 16)
- `alpha`: LoRA alpha scaling (default: 16)
- `target_modules`: Which layers to apply LoRA to

### Training Settings
- `batch_size`: Per-device batch size
- `gradient_accumulation_steps`: Gradient accumulation
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs

## Output

The training script saves:
- **Checkpoints**: Regular checkpoints during training (`outputs/checkpoint-*/`)
- **Final Model**: LoRA adapters (`outputs/final_model/`)
- **Merged Model**: Optional full merged model (`outputs/merged_model/`)

## Inference

After training, load your fine-tuned model:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./outputs/final_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Generate text
inputs = tokenizer("### Instruction:\nWhat is machine learning?\n\n### Response:\n", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Memory Usage

Approximate VRAM usage with default settings:
- **4-bit + LoRA**: ~10-12 GB
- **Full fine-tuning**: ~40+ GB (not recommended)

Tips to reduce memory:
- Decrease `max_seq_length`
- Decrease `batch_size`
- Increase `gradient_accumulation_steps`

## Troubleshooting

### "No module named 'unsloth'"
- Ensure you're using Python 3.10+
- Reinstall: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`

### CUDA Out of Memory
- Reduce batch size
- Enable gradient checkpointing (already enabled by default)
- Use smaller `max_seq_length`

### Slow Training
- Ensure you're using a GPU
- Check that CUDA is properly installed
- Verify Unsloth is properly installed

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Llama-3 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## License

Please refer to Meta's Llama-3 license for model usage terms.
