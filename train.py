import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math
import argparse
import json
from tqdm import tqdm
from utils.xor_transformer import XORTransformer, set_seed
import sys
from datetime import datetime
import time

if __name__ == "__main__":
    # Start total execution time measurement
    total_start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', type=str)
    args = parser.parse_args()

    # Parse conditions from JSON string
    conditions = json.loads(args.conditions)

    # Create filename based on conditions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/xor_results_{timestamp}_Dk{conditions['D_k']}_w{conditions['weights']}_b{conditions['bias']}_pe{conditions['positional_encoding']}_sm{conditions['softmax']}_ln{conditions['layer_norm']}.txt"

    # XOR data preparation
    inputs = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Redirect stdout to both file and console
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        print(f"Training with conditions: {conditions}\n")
        
        for seed in range(10):
            # Start seed execution time measurement
            seed_start_time = time.time()
            
            print(f"\n{'='*50}")
            print(f"Starting training with seed {seed}")
            print(f"{'='*50}\n")
            
            # Set random seed
            set_seed(seed)

            # initialize model
            model = XORTransformer(conditions)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # training
            model.train()
            for epoch in tqdm(range(1000), desc="Training"):
                optimizer.zero_grad()
                outputs = []
                for x in inputs:
                    prob, pred = model.forward(x)
                    outputs.append(prob)

                outputs = torch.cat(outputs)
                loss = torch.nn.BCELoss()(outputs, targets)
                loss.backward()
                optimizer.step()
                
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # test
            print("\nTest:")
            model.eval()
            
            outputs = []
            predictions = []
            for x in inputs:
                prob, pred = model.forward(x)
                outputs.append(prob)
                predictions.append(pred)
                print(f"Input: {x.tolist()}, Prob: {prob.item():.4f}, Output: {pred}")

            outputs = torch.cat(outputs)
            predictions = torch.tensor(predictions)
            
            # Calculate seed execution time
            seed_end_time = time.time()
            seed_execution_time = seed_end_time - seed_start_time
            
            if torch.all(predictions == targets):
                print(f"\nseed {seed}: SUCCESS")
                print(f"Execution time for seed {seed}: {seed_execution_time:.2f} seconds")
                break
            else:
                print(f"\nseed {seed}: FAILED")
                print(f"Execution time for seed {seed}: {seed_execution_time:.2f} seconds")
            
            # Flush the file to ensure all output is written
            f.flush()
            
            # Also print to console
            sys.stdout = original_stdout
            print(f"Seed {seed} completed - {'SUCCESS' if torch.all(predictions == targets) else 'FAILED'} (Time: {seed_execution_time:.2f}s)")
            sys.stdout = f

    # Calculate total execution time
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    # Restore stdout
    sys.stdout = original_stdout
    print(f"\nResults have been saved to: {filename}")
    print(f"Total execution time: {total_execution_time:.2f} seconds")