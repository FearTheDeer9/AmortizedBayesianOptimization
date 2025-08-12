#!/usr/bin/env python3
"""
Monitor training progress by reading the output file.
"""

import time
from pathlib import Path

def monitor_training():
    output_file = Path("numerical_sort_training_output.txt")
    
    if not output_file.exists():
        print("Training output file not found yet...")
        return
    
    print("="*60)
    print("MONITORING TRAINING PROGRESS")
    print("="*60)
    
    # Read last 50 lines
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    # Find epoch lines
    epoch_lines = []
    for i, line in enumerate(lines):
        if "Epoch" in line and "/" in line:
            # Get this line and the next few
            epoch_info = []
            for j in range(5):
                if i+j < len(lines):
                    epoch_info.append(lines[i+j].strip())
            epoch_lines.append("\n".join(epoch_info))
    
    # Show last 5 epochs
    print("\nLast 5 epochs:")
    print("-" * 40)
    for epoch_info in epoch_lines[-5:]:
        print(epoch_info)
        print()
    
    # Check for completion
    if any("TRAINING COMPLETE" in line for line in lines):
        print("\n✓ Training completed!")
        
        # Find final results
        for i, line in enumerate(lines):
            if "Best validation accuracy" in line:
                print(line.strip())
            if "Final X4 accuracy" in line:
                print(line.strip())
    
    # Check for errors
    if any("Traceback" in line for line in lines[-20:]):
        print("\n⚠️ Error detected in training!")
        print("Last 10 lines:")
        for line in lines[-10:]:
            print(line.strip())

if __name__ == "__main__":
    monitor_training()