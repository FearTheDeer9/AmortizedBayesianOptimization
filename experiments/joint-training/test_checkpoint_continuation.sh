#!/bin/bash

echo "Testing Joint Training Checkpoint Continuation"
echo "============================================="
echo ""
echo "This will run 5 phases (10 minutes total, 2 minutes each):"
echo "  Phase 1: Surrogate from scratch (2 min)"
echo "  Phase 2: Policy from scratch (using Phase 1 surrogate, 2 min)"
echo "  Phase 3: Surrogate continuing from Phase 1 (2 min)"
echo "  Phase 4: Policy continuing from Phase 2 (using Phase 3 surrogate, 2 min)"
echo "  Phase 5: Surrogate continuing from Phase 3 (2 min)"
echo ""
echo "Expected checkpoints:"
echo "  - surrogate_phase_1.pkl"
echo "  - policy_phase_2.pkl" 
echo "  - surrogate_phase_3.pkl"
echo "  - policy_phase_4.pkl"
echo "  - surrogate_phase_5.pkl"
echo ""

# Clean up any existing test checkpoints
rm -rf experiments/joint-training/checkpoints/checkpoint_test

echo "Starting training..."
python experiments/joint-training/train_joint_orchestrator.py \
    --config experiments/joint-training/configs/checkpoint_test.json

echo ""
echo "Training complete. Checking results..."
echo ""

CHECKPOINT_DIR="experiments/joint-training/checkpoints/checkpoint_test"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Generated checkpoints:"
    ls -la "$CHECKPOINT_DIR"/*.pkl 2>/dev/null || echo "No .pkl files found"
    
    echo ""
    echo "Analyzing checkpoint metadata to verify continuation..."
    echo ""
    
    # Use Python to check checkpoint metadata
    python -c "
import pickle
import sys
from pathlib import Path

checkpoint_dir = Path('$CHECKPOINT_DIR')

for checkpoint_file in sorted(checkpoint_dir.glob('*.pkl')):
    print(f'=== {checkpoint_file.name} ===')
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        
        metadata = data.get('metadata', {})
        print(f'  Model type: {data.get(\"model_type\", \"unknown\")}')
        print(f'  Step/Episode: {metadata.get(\"step\", metadata.get(\"episode\", \"unknown\"))}')
        
        if 'surrogate' in checkpoint_file.name:
            print(f'  F1 score: {metadata.get(\"avg_f1\", \"unknown\")}')
        elif 'policy' in checkpoint_file.name:
            print(f'  Episodes completed: {metadata.get(\"episode\", \"unknown\")}')
        
        print('')
    except Exception as e:
        print(f'  Error reading: {e}')
        print('')
    "
    
else
    echo "ERROR: Checkpoint directory not found!"
fi

echo ""
echo "To verify continuation manually, look for:"
echo "1. Phase 3 surrogate should have higher step number than Phase 1"  
echo "2. Phase 2 policy should show it loaded a surrogate checkpoint"
echo "3. Check the training logs for 'Resuming from:' messages"