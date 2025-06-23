#!/bin/bash
#SBATCH --job-name=acbo_train_acquisition
#SBATCH --partition=training          # Use training partition for taught students
#SBATCH --qos=training               # Quality of service for training
#SBATCH --gres=gpu:a100:1            # Request 1 A100 GPU (80GB) for RL training
#SBATCH --cpus-per-task=16           # 16 CPU cores for environment simulation
#SBATCH --mem=64G                    # 64GB RAM for experience replay
#SBATCH --time=72:00:00              # 72 hour time limit for RL training
#SBATCH --output=/vol/bitbucket/%u/causal_bayes_opt/logs/training/train_acquisition_%j.out
#SBATCH --error=/vol/bitbucket/%u/causal_bayes_opt/logs/training/train_acquisition_%j.err

# Job information
echo "=================================================="
echo "ACBO Acquisition Model Training Job (GRPO)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=================================================="

# Configuration - Edit these parameters as needed
SURROGATE_MODEL=${SURROGATE_MODEL:-"/vol/bitbucket/$USER/causal_bayes_opt/results/models/surrogate/best_model.pkl"}
DATA_DIR=${DATA_DIR:-"/vol/bitbucket/$USER/causal_bayes_opt/data/processed"}
OUTPUT_DIR=${OUTPUT_DIR:-"/vol/bitbucket/$USER/causal_bayes_opt/results/models/acquisition"}

# Training configuration
BC_EPOCHS=${BC_EPOCHS:-50}           # Behavioral cloning epochs
GRPO_EPISODES=${GRPO_EPISODES:-1000} # GRPO training episodes
BATCH_SIZE=${BATCH_SIZE:-64}         # GRPO batch size
BC_LEARNING_RATE=${BC_LEARNING_RATE:-1e-3}
GRPO_LEARNING_RATE=${GRPO_LEARNING_RATE:-1e-4}

# GRPO specific settings
GROUP_SIZE=${GROUP_SIZE:-64}         # GRPO group size
ENTROPY_WEIGHT=${ENTROPY_WEIGHT:-0.01}
VALUE_WEIGHT=${VALUE_WEIGHT:-0.5}
CLIP_RATIO=${CLIP_RATIO:-0.2}

# Reward configuration
REWARD_TYPE=${REWARD_TYPE:-"verifiable"}  # verifiable/hybrid
TARGET_WEIGHT=${TARGET_WEIGHT:-2.0}
PARENT_WEIGHT=${PARENT_WEIGHT:-1.0}
EXPLORATION_WEIGHT=${EXPLORATION_WEIGHT:-0.5}

# Setup environment
PROJECT_DIR="/vol/bitbucket/$USER/causal_bayes_opt"
cd "$PROJECT_DIR"

# Activate environment
echo "Activating environment..."
source activate_env.sh

# Check GPU availability
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Create output directories
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/plots"
mkdir -p "$OUTPUT_DIR/evaluation"

# Validate prerequisites
echo "Validating prerequisites..."

# Check surrogate model
if [ ! -f "$SURROGATE_MODEL" ]; then
    echo "❌ Surrogate model not found at: $SURROGATE_MODEL"
    echo "Please train surrogate model first using train_surrogate.sbatch"
    exit 1
fi

# Check training data
if [ ! -d "$DATA_DIR/splits" ]; then
    echo "❌ Split training data not found at: $DATA_DIR/splits"
    echo "Please run data preprocessing first."
    exit 1
fi

echo "✅ Prerequisites validated"
echo ""

echo "Starting acquisition model training with configuration:"
echo "  Surrogate model: $SURROGATE_MODEL"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Behavioral Cloning Phase:"
echo "  BC epochs: $BC_EPOCHS"
echo "  BC learning rate: $BC_LEARNING_RATE"
echo ""
echo "GRPO Training Phase:"
echo "  GRPO episodes: $GRPO_EPISODES"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $GRPO_LEARNING_RATE"
echo "  Group size: $GROUP_SIZE"
echo "  Entropy weight: $ENTROPY_WEIGHT"
echo "  Value weight: $VALUE_WEIGHT"
echo "  Clip ratio: $CLIP_RATIO"
echo ""
echo "Reward Configuration:"
echo "  Reward type: $REWARD_TYPE"
echo "  Target weight: $TARGET_WEIGHT"
echo "  Parent weight: $PARENT_WEIGHT"
echo "  Exploration weight: $EXPLORATION_WEIGHT"
echo ""

# Check for existing checkpoint to resume from
LATEST_CHECKPOINT=$(find "$OUTPUT_DIR/checkpoints" -name "grpo_checkpoint_*.pkl" 2>/dev/null | sort -V | tail -1)
RESUME_ARG=""
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found GRPO checkpoint: $LATEST_CHECKPOINT"
    RESUME_ARG="--resume $LATEST_CHECKPOINT"
fi

# Build training command
CMD="python -m src.causal_bayes_opt.training.acquisition_training"
CMD="$CMD --surrogate-model $SURROGATE_MODEL"
CMD="$CMD --data-dir $DATA_DIR/splits"
CMD="$CMD --output-dir $OUTPUT_DIR"

# Behavioral cloning phase
CMD="$CMD --bc-epochs $BC_EPOCHS"
CMD="$CMD --bc-learning-rate $BC_LEARNING_RATE"

# GRPO phase
CMD="$CMD --grpo-episodes $GRPO_EPISODES"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --grpo-learning-rate $GRPO_LEARNING_RATE"
CMD="$CMD --group-size $GROUP_SIZE"
CMD="$CMD --entropy-weight $ENTROPY_WEIGHT"
CMD="$CMD --value-weight $VALUE_WEIGHT"
CMD="$CMD --clip-ratio $CLIP_RATIO"

# Reward configuration
CMD="$CMD --reward-type $REWARD_TYPE"
CMD="$CMD --target-weight $TARGET_WEIGHT"
CMD="$CMD --parent-weight $PARENT_WEIGHT"
CMD="$CMD --exploration-weight $EXPLORATION_WEIGHT"

# Other settings
CMD="$CMD --device gpu"
CMD="$CMD --log-level INFO"
CMD="$CMD --save-interval 100"

if [ -n "$RESUME_ARG" ]; then
    CMD="$CMD $RESUME_ARG"
fi

echo "Executing: $CMD"
echo "=================================================="

# Start training with error handling
eval $CMD
TRAINING_EXIT_CODE=$?

# Post-training evaluation
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Running comprehensive evaluation..."
    
    # Find the best policy model
    BEST_POLICY=$(find "$OUTPUT_DIR" -name "best_policy.pkl" | head -1)
    if [ -n "$BEST_POLICY" ]; then
        echo "Evaluating best policy: $BEST_POLICY"
        
        # Run comprehensive evaluation
        python -c "
import pickle
import numpy as np
from pathlib import Path
from src.causal_bayes_opt.training.acquisition_training import evaluate_acquisition_policy

# Load best policy
with open('$BEST_POLICY', 'rb') as f:
    policy_state = pickle.load(f)

print('🎯 Running policy evaluation...')

# Evaluate on test environments
test_results = evaluate_acquisition_policy(
    policy_state,
    '$SURROGATE_MODEL',
    '$DATA_DIR/splits/test_data.pkl',
    n_episodes=50,
    detailed=True
)

print('📊 Final Policy Evaluation Results:')
print(f'Average Reward: {test_results[\"avg_reward\"]:.3f} ± {test_results[\"reward_std\"]:.3f}')
print(f'Success Rate: {test_results[\"success_rate\"]:.1%}')
print(f'Target Improvement: {test_results[\"avg_target_improvement\"]:.3f}')
print(f'Parent Discovery Rate: {test_results[\"parent_discovery_rate\"]:.1%}')
print(f'Exploration Diversity: {test_results[\"exploration_diversity\"]:.3f}')

# Save detailed results
results_path = Path('$OUTPUT_DIR/evaluation/final_evaluation.json')
results_path.parent.mkdir(exist_ok=True)
import json
with open(results_path, 'w') as f:
    json.dump(test_results, f, indent=2, default=str)

print(f'💾 Detailed results saved to: {results_path}')
"
        
        # Compare with baseline (PARENT_SCALE)
        echo ""
        echo "Running baseline comparison..."
        python -c "
from src.causal_bayes_opt.integration.parent_scale_bridge import run_parent_scale_baseline

# Run PARENT_SCALE baseline on same test problems
baseline_results = run_parent_scale_baseline('$DATA_DIR/splits/test_data.pkl')

print('📈 Baseline Comparison (vs PARENT_SCALE):')
print(f'ACBO Policy vs Baseline Improvement: {test_results[\"avg_target_improvement\"]} vs {baseline_results[\"avg_improvement\"]}')
improvement_ratio = test_results['avg_target_improvement'] / baseline_results['avg_improvement']
print(f'Relative Performance: {improvement_ratio:.2f}x')

if improvement_ratio > 1.0:
    print('🎉 ACBO policy outperforms PARENT_SCALE baseline!')
else:
    print('⚠️ ACBO policy underperforms baseline - consider retraining')
"
        
    else
        echo "⚠️ Best policy not found for evaluation"
    fi
    
    # Generate training plots
    echo ""
    echo "Generating training visualization..."
    python -c "
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training logs
bc_log = Path('$OUTPUT_DIR/logs/bc_training.json')
grpo_log = Path('$OUTPUT_DIR/logs/grpo_training.json')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Behavioral cloning plots
if bc_log.exists():
    with open(bc_log, 'r') as f:
        bc_logs = [json.loads(line) for line in f]
    
    epochs = [log['epoch'] for log in bc_logs]
    bc_loss = [log['loss'] for log in bc_logs]
    bc_accuracy = [log['accuracy'] for log in bc_logs]
    
    axes[0, 0].plot(epochs, bc_loss)
    axes[0, 0].set_title('Behavioral Cloning Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    axes[0, 1].plot(epochs, bc_accuracy)
    axes[0, 1].set_title('Behavioral Cloning Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')

# GRPO training plots
if grpo_log.exists():
    with open(grpo_log, 'r') as f:
        grpo_logs = [json.loads(line) for line in f]
    
    episodes = [log['episode'] for log in grpo_logs]
    rewards = [log['avg_reward'] for log in grpo_logs]
    policy_loss = [log['policy_loss'] for log in grpo_logs]
    
    axes[1, 0].plot(episodes, rewards)
    axes[1, 0].set_title('GRPO Average Reward')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Reward')
    
    axes[1, 1].plot(episodes, policy_loss)
    axes[1, 1].set_title('GRPO Policy Loss')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/plots/training_progress.png', dpi=300, bbox_inches='tight')
print('📈 Training plots saved to $OUTPUT_DIR/plots/')

# Create reward component analysis
if grpo_log.exists():
    target_rewards = [log.get('target_reward', 0) for log in grpo_logs]
    parent_rewards = [log.get('parent_reward', 0) for log in grpo_logs]
    exploration_rewards = [log.get('exploration_reward', 0) for log in grpo_logs]
    
    plt.figure(figsize=(12, 4))
    plt.plot(episodes, target_rewards, label='Target Improvement', alpha=0.7)
    plt.plot(episodes, parent_rewards, label='Parent Discovery', alpha=0.7)
    plt.plot(episodes, exploration_rewards, label='Exploration', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Reward Component')
    plt.title('Reward Component Analysis')
    plt.legend()
    plt.savefig('$OUTPUT_DIR/plots/reward_components.png', dpi=300, bbox_inches='tight')
"
    
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    
    # Attempt to diagnose failure
    echo ""
    echo "Diagnosing failure..."
    
    # Check for common issues
    if [ ! -f "$SURROGATE_MODEL" ]; then
        echo "❌ Issue: Surrogate model missing"
    fi
    
    # Check GPU memory
    echo "GPU Memory Status:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    
    # Check disk space
    echo "Disk Space:"
    df -h /vol/bitbucket/$USER/
fi

# Print job statistics
echo ""
echo "=================================================="
echo "Job Statistics:"
echo "End time: $(date)"
echo "Duration: $SECONDS seconds"
echo "Exit code: $TRAINING_EXIT_CODE"

# Final GPU status
echo ""
echo "Final GPU utilization:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Archive logs and create summary
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Archiving logs and creating summary..."
    
    # Copy Slurm logs to output directory
    cp "/vol/bitbucket/$USER/causal_bayes_opt/logs/training/train_acquisition_${SLURM_JOB_ID}.out" "$OUTPUT_DIR/logs/" 2>/dev/null || true
    cp "/vol/bitbucket/$USER/causal_bayes_opt/logs/training/train_acquisition_${SLURM_JOB_ID}.err" "$OUTPUT_DIR/logs/" 2>/dev/null || true
    
    # Create training summary
    cat > "$OUTPUT_DIR/training_summary.txt" << EOF
ACBO Acquisition Model Training Summary
======================================
Job ID: $SLURM_JOB_ID
Node: $SLURMD_NODENAME
Start time: $(date -d "@$(($(date +%s) - $SECONDS))")
End time: $(date)
Duration: $SECONDS seconds

Configuration:
- Surrogate model: $SURROGATE_MODEL
- BC epochs: $BC_EPOCHS
- GRPO episodes: $GRPO_EPISODES
- Batch size: $BATCH_SIZE
- Group size: $GROUP_SIZE
- Reward type: $REWARD_TYPE

GPU: $CUDA_VISIBLE_DEVICES
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)

Status: SUCCESS

Next Steps:
1. Review evaluation results in $OUTPUT_DIR/evaluation/
2. Check training plots in $OUTPUT_DIR/plots/
3. Test the trained policy with: python scripts/test_acbo_policy.py
EOF

    echo "✅ Training summary created: $OUTPUT_DIR/training_summary.txt"
fi

exit $TRAINING_EXIT_CODE