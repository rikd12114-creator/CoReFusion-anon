import pandas as pd
import numpy as np
import random
import csv
import os
from datetime import datetime

OUTPUT_DIR = "experiments/smell_as_noise_refactoring/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_mock_data(num_samples=20):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"retention_rate_mock_{timestamp}.csv")
    
    fieldnames = [
        'sample_id', 'condition', 'noise_level', 'trial_id', 
        'original_name', 'target_name', 'predicted_name', 
        'is_restored', 'is_refactored', 'target_masked'
    ]
    
    noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    trials = 3
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(num_samples):
            sample_id = f"mock_{i}"
            original_name = "data"
            target_name = "userData"
            
            for noise in noise_levels:
                for t in range(trials):
                    # Simulate results based on hypothesis:
                    # 1. Clean code (condition='clean') is more stable (high retention)
                    # 2. Smelly code (condition='smelly') is less stable
                    # 3. Refactoring peaks at mid-noise (e.g., 0.3-0.5)
                    
                    # Condition: Clean
                    is_masked = random.random() < noise
                    if is_masked:
                        # Clean code restoration probability drops with noise
                        restored_prob = max(0.2, 1.0 - noise * 0.8) 
                    else:
                        restored_prob = 1.0
                    
                    is_restored = random.random() < restored_prob
                    writer.writerow({
                        'sample_id': sample_id, 'condition': 'clean', 'noise_level': noise, 'trial_id': t,
                        'original_name': target_name, 'target_name': target_name, 'predicted_name': target_name if is_restored else "other",
                        'is_restored': is_restored, 'is_refactored': False, 'target_masked': is_masked
                    })
                    
                    # Condition: Smelly
                    is_masked = random.random() < noise
                    if is_masked:
                        # Smelly code restoration (keeping smell) drops faster
                        restored_prob = max(0.1, 0.9 - noise * 1.0)
                        # Refactoring probability peaks at 0.5
                        refactor_prob = 0.4 * np.exp(-(noise - 0.4)**2 / 0.1)
                    else:
                        restored_prob = 1.0
                        refactor_prob = 0.0
                        
                    outcome = random.random()
                    if outcome < restored_prob:
                        is_restored = True
                        is_refactored = False
                        pred = original_name
                    elif outcome < restored_prob + refactor_prob:
                        is_restored = False
                        is_refactored = True
                        pred = target_name
                    else:
                        is_restored = False
                        is_refactored = False
                        pred = "random_noise"
                        
                    writer.writerow({
                        'sample_id': sample_id, 'condition': 'smelly', 'noise_level': noise, 'trial_id': t,
                        'original_name': original_name, 'target_name': target_name, 'predicted_name': pred,
                        'is_restored': is_restored, 'is_refactored': is_refactored, 'target_masked': is_masked
                    })
    
    print(f"Generated mock data at {filename}")
    return filename

if __name__ == "__main__":
    generate_mock_data()
