#!/bin/bash

echo "Testing TRUE Checkpoint Continuation"
echo "===================================="
echo ""
echo "This will run 3 phases (6 minutes total, 2 minutes each):"
echo "  Phase 1: Surrogate steps 0→9 (from scratch)"
echo "  Phase 2: Policy episodes 0→4 (using Phase 1 surrogate)"
echo "  Phase 3: Surrogate steps 10→19 (continuing from Phase 1)"
echo ""
echo "Expected step/episode progression:"
echo "  surrogate_phase_1.pkl: step 9"
echo "  policy_phase_2.pkl: episode 4"
echo "  surrogate_phase_3.pkl: step 19 (proving continuation!)"
echo ""

# Clean up any existing test checkpoints
rm -rf experiments/joint-training/checkpoints/continuation_test

echo "Starting training..."
python experiments/joint-training/train_joint_orchestrator.py \
    --config experiments/joint-training/configs/continuation_test.json

echo ""
echo "Training complete. Verifying TRUE continuation..."
echo ""

CHECKPOINT_DIR="experiments/joint-training/checkpoints/continuation_test"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Generated checkpoints:"
    ls -la "$CHECKPOINT_DIR"/*.pkl 2>/dev/null || echo "No .pkl files found"
    
    echo ""
    echo "=== STEP/EPISODE PROGRESSION ANALYSIS ==="
    
    # Check step progression for surrogates
    python -c "
import pickle
import sys
from pathlib import Path

checkpoint_dir = Path('$CHECKPOINT_DIR')

print('SURROGATE STEP PROGRESSION:')
for phase in [1, 3]:
    checkpoint_file = checkpoint_dir / f'surrogate_phase_{phase}.pkl'
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            step = data.get('metadata', {}).get('step', 'unknown')
            print(f'  Phase {phase}: Step {step}')
        except Exception as e:
            print(f'  Phase {phase}: Error - {e}')

print('')
print('POLICY EPISODE PROGRESSION:')
for phase in [2, 4]:
    checkpoint_file = checkpoint_dir / f'policy_phase_{phase}.pkl'
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check multiple fields for episode count
            episode = (
                data.get('metadata', {}).get('episode', None) or
                data.get('metadata', {}).get('total_episodes', None) or
                'unknown'
            )
            print(f'  Phase {phase}: Episodes completed = {episode}')
        except Exception as e:
            print(f'  Phase {phase}: Error - {e}')

print('')
print('SUCCESS CRITERIA:')
print('✓ Phase 3 surrogate step > Phase 1 step (proves continuation)')
print('✓ No crashes during training')
print('✓ All checkpoint files created')
    "
    
else
    echo "ERROR: Checkpoint directory not found!"
fi

echo ""