#!/bin/bash
#SBATCH --job-name=acbo_train_surrogate
#SBATCH --partition=training          # Use training partition for taught students  
#SBATCH --qos=training               # Quality of service for training
#SBATCH --gres=gpu:a100:1            # Request 1 A100 GPU (80GB) for large models
#SBATCH --cpus-per-task=16           # 16 CPU cores for data loading
#SBATCH --mem=64G                    # 64GB RAM for training
#SBATCH --time=48:00:00              # 48 hour time limit for training
#SBATCH --output=/vol/bitbucket/%u/causal_bayes_opt/logs/training/train_surrogate_%j.out
#SBATCH --error=/vol/bitbucket/%u/causal_bayes_opt/logs/training/train_surrogate_%j.err

# Job information
echo "=================================================="
echo "ACBO Surrogate Model Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=================================================="

# Configuration - Edit these parameters as needed
DATA_DIR=${DATA_DIR:-"/vol/bitbucket/$USER/causal_bayes_opt/data/processed"}
OUTPUT_DIR=${OUTPUT_DIR:-"/vol/bitbucket/$USER/causal_bayes_opt/results/models/surrogate"}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
CURRICULUM=${CURRICULUM:-"progressive"}  # progressive/mixed/fixed
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10}

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

# Check for training data
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Training data not found at: $DATA_DIR"
    echo "Please run data collection and preprocessing first."
    exit 1
fi

# Check for split data
SPLIT_DIR="$DATA_DIR/splits"
if [ ! -d "$SPLIT_DIR" ]; then
    echo "Split data not found, creating splits..."
    python scripts/split_dataset.py "$DATA_DIR" --curriculum --stratify difficulty graph_type
fi

echo "Starting surrogate model training with configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Curriculum: $CURRICULUM"
echo "  Checkpoint interval: $CHECKPOINT_INTERVAL"
echo ""

# Check for existing checkpoint to resume from
LATEST_CHECKPOINT=$(find "$OUTPUT_DIR/checkpoints" -name "checkpoint_*.pkl" 2>/dev/null | sort -V | tail -1)
RESUME_ARG=""
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    echo "Resume training? (y/n)"
    # In batch mode, assume yes for resume
    RESUME_ARG="--resume $LATEST_CHECKPOINT"
fi

# Build training command
CMD="python -m src.causal_bayes_opt.training.surrogate_trainer"
CMD="$CMD --data-dir $SPLIT_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --curriculum $CURRICULUM"
CMD="$CMD --checkpoint-interval $CHECKPOINT_INTERVAL"
CMD="$CMD --device gpu"
CMD="$CMD --log-level INFO"

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
    echo "Running evaluation..."
    
    # Find the best model
    BEST_MODEL=$(find "$OUTPUT_DIR" -name "best_model.pkl" | head -1)
    if [ -n "$BEST_MODEL" ]; then
        echo "Evaluating best model: $BEST_MODEL"
        
        # Run evaluation
        python -c "
import pickle
from src.causal_bayes_opt.training.surrogate_training import validate_surrogate_performance

# Load model and evaluate
with open('$BEST_MODEL', 'rb') as f:
    model_state = pickle.load(f)

# Run validation on test set
results = validate_surrogate_performance(
    model_state, 
    '$SPLIT_DIR/test_data.pkl',
    detailed=True
)

print('📊 Final Evaluation Results:')
print(f'Test Accuracy: {results[\"accuracy\"]:.3f}')
print(f'Test Loss: {results[\"loss\"]:.3f}')
print(f'F1 Score: {results.get(\"f1_score\", \"N/A\")}')
"
    else
        echo "⚠️ Best model not found for evaluation"
    fi
    
    # Generate training plots
    echo "Generating training plots..."
    python -c "
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training logs
log_file = Path('$OUTPUT_DIR/logs/training_log.json')
if log_file.exists():
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f]
    
    # Extract metrics
    epochs = [log['epoch'] for log in logs]
    train_loss = [log['train_loss'] for log in logs]
    val_loss = [log['val_loss'] for log in logs]
    
    # Create plots
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    
    plt.subplot(1, 2, 2)
    if 'accuracy' in logs[0]:
        train_acc = [log['train_accuracy'] for log in logs]
        val_acc = [log['val_accuracy'] for log in logs]
        plt.plot(epochs, train_acc, label='Train Accuracy')
        plt.plot(epochs, val_acc, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Progress')
    
    plt.tight_layout()
    plt.savefig('$OUTPUT_DIR/plots/training_progress.png', dpi=300, bbox_inches='tight')
    print('📈 Training plots saved to $OUTPUT_DIR/plots/')
else:
    print('⚠️ Training log not found')
"
    
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
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
echo "Final GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Archive logs and cleanup
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Archiving logs..."
    
    # Copy Slurm logs to output directory
    cp "/vol/bitbucket/$USER/causal_bayes_opt/logs/training/train_surrogate_${SLURM_JOB_ID}.out" "$OUTPUT_DIR/logs/" 2>/dev/null || true
    cp "/vol/bitbucket/$USER/causal_bayes_opt/logs/training/train_surrogate_${SLURM_JOB_ID}.err" "$OUTPUT_DIR/logs/" 2>/dev/null || true
    
    # Create training summary
    echo "Creating training summary..."
    cat > "$OUTPUT_DIR/training_summary.txt" << EOF
ACBO Surrogate Model Training Summary
====================================
Job ID: $SLURM_JOB_ID
Node: $SLURMD_NODENAME
Start time: $(date -d "@$(($(date +%s) - $SECONDS))")
End time: $(date)
Duration: $SECONDS seconds

Configuration:
- Data directory: $DATA_DIR
- Epochs: $EPOCHS
- Batch size: $BATCH_SIZE
- Learning rate: $LEARNING_RATE
- Curriculum: $CURRICULUM

GPU: $CUDA_VISIBLE_DEVICES
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)

Status: SUCCESS
EOF
fi

exit $TRAINING_EXIT_CODE