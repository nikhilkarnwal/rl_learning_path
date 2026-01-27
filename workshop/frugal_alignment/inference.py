"""
Inference script for fine-tuned Llama-3-8B model
"""

import torch
from unsloth import FastLanguageModel
import argparse


def load_model(model_path: str, max_seq_length: int = 2048):
    """Load fine-tuned model for inference."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Generate response for given instruction."""
    # Format prompt
    prompt = f"""### Instruction:
{instruction}

### Response:
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Llama-3")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/final_model",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Instruction/prompt for the model"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.model_path)
    
    # Generate response
    print(f"\nInstruction: {args.instruction}\n")
    print("Generating response...")
    response = generate_response(
        model,
        tokenizer,
        args.instruction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    print(f"\nResponse:\n{response}\n")


if __name__ == "__main__":
    main()
