#!/usr/bin/env python3
import argparse
import random
import json
import os

def read_prompts_from_file(file_path):
    """Read prompts from a text file, one prompt per line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        # Strip whitespace and filter out empty lines
        return [line.strip() for line in f if line.strip()]

def select_random_prompts(prompts, n, seed=None):
    """Randomly select n prompts from the list."""
    if seed is not None:
        random.seed(seed)
    
    # Make sure n is not larger than the number of available prompts
    n = min(n, len(prompts))
    
    return random.sample(prompts, n)

def save_to_json(prompts, output_file):
    """Save the prompts to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=4)
    print(f"Saved {len(prompts)} prompts to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Select random prompts from a text file and save to JSON.')
    parser.add_argument('--input', type=str, required=True, help='Input text file with prompts (one per line)')
    parser.add_argument('--output', type=str, default='captions_videos.json', help='Output JSON file')
    parser.add_argument('--count', type=int, default=50, help='Number of prompts to select')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return
    
    # Read all prompts from the input file
    all_prompts = read_prompts_from_file(args.input)
    print(f"Read {len(all_prompts)} prompts from {args.input}")
    
    # Select random prompts
    selected_prompts = select_random_prompts(all_prompts, args.count, args.seed)
    
    # Save to JSON
    save_to_json(selected_prompts, args.output)
    
    # Print example
    print(f"\nExample of the first 5 selected prompts:")
    for i, prompt in enumerate(selected_prompts[:5]):
        print(f"{i+1}. {prompt}")

if __name__ == "__main__":
    main()