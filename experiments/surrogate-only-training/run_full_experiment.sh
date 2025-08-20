#!/bin/bash
# Full training and evaluation pipeline for 100-variable surrogate model

set -e  # Exit on error

# Configuration
MODEL_SIZE=${1:-large}  # Default to large
STAGES=${2:-1-5}        # Default to all stages
SEED=${3:-42}           # Random seed

echo "=========================================="
echo "SURROGATE MODEL TRAINING & EVALUATION"
echo "=========================================="
echo "Model size: $MODEL_SIZE"
echo "Stages: $STAGES"
echo "Seed: $SEED"
echo ""

# Create directories
mkdir -p logs
mkdir -p results/plots

# Step 1: Profile model sizes (optional, comment out if already done)
echo "Step 1: Profiling model sizes..."
python scripts/profile_model_sizes.py > logs/profiling_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "Profiling complete. Check logs for recommendations."
echo ""

# Step 2: Start training
echo "Step 2: Starting training..."
LOG_FILE="logs/training_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S).log"

if [ "$STAGES" == "1-5" ]; then
    echo "Running full curriculum (this will take 2-4 days)..."
    nohup python scripts/train_scaled_curriculum.py \
        --model-size $MODEL_SIZE \
        --stages $STAGES \
        --seed $SEED \
        > "$LOG_FILE" 2>&1 &
    
    TRAIN_PID=$!
    echo "Training started in background (PID: $TRAIN_PID)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Wait for at least stage 1 to complete
    echo "Waiting for Stage 1 to complete..."
    while [ ! -f checkpoints/runs/run_100var_*/stage_1_complete.pkl ]; do
        sleep 60
        echo -n "."
    done
    echo " Stage 1 complete!"
    
else
    # Run in foreground for shorter training
    python scripts/train_scaled_curriculum.py \
        --model-size $MODEL_SIZE \
        --stages $STAGES \
        --seed $SEED \
        | tee "$LOG_FILE"
fi

# Step 3: Find the run directory
echo ""
echo "Step 3: Locating checkpoint directory..."
CHECKPOINT_DIR=$(ls -td checkpoints/runs/run_100var_* | head -1)
echo "Found: $CHECKPOINT_DIR"

# Step 4: Run evaluation
echo ""
echo "Step 4: Running evaluation..."
EVAL_LOG="logs/evaluation_$(date +%Y%m%d_%H%M%S).log"

python scripts/evaluate_scaled_models.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --interventions 30 \
    --test-sizes "5,10,20,30,40,50,60,70,80,90,100" \
    --test-structures all \
    --output-plots \
    --seed 999 \
    > "$EVAL_LOG" 2>&1

echo "Evaluation complete. Log: $EVAL_LOG"

# Step 5: Generate summary
echo ""
echo "Step 5: Generating summary..."
echo "=========================================="
echo "TRAINING SUMMARY"
echo "=========================================="

# Extract key metrics
if [ -f "$CHECKPOINT_DIR/training_metrics.json" ]; then
    echo "Training metrics:"
    python -c "
import json
with open('$CHECKPOINT_DIR/training_metrics.json') as f:
    metrics = json.load(f)
    if metrics:
        completed = sum(1 for m in metrics if m.get('success', False))
        total_interventions = sum(m.get('interventions', 0) for m in metrics)
        max_vars = max(m['scm_config']['num_vars'] for m in metrics if m.get('success', False))
        print(f'  Completed SCMs: {completed}/{len(metrics)}')
        print(f'  Total interventions: {total_interventions}')
        print(f'  Largest solved: {max_vars} variables')
"
fi

echo ""
echo "=========================================="
echo "EVALUATION SUMMARY"
echo "=========================================="

# Extract evaluation results
LATEST_EVAL=$(ls -t results/scaled_evaluation_*.json | head -1)
if [ -f "$LATEST_EVAL" ]; then
    echo "Evaluation results: $LATEST_EVAL"
    python -c "
import json
with open('$LATEST_EVAL') as f:
    results = json.load(f)
    for model_name, data in results.items():
        if 'summary' in data:
            s = data['summary']
            print(f'  {model_name}:')
            print(f'    Mean F1: {s[\"mean_f1\"]:.3f} ± {s[\"std_f1\"]:.3f}')
            print(f'    Range: [{s[\"min_f1\"]:.3f}, {s[\"max_f1\"]:.3f}]')
"
fi

echo ""
echo "Plots saved to: results/plots_*/"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "=========================================="
echo "EXPERIMENT COMPLETE"
echo "=========================================="