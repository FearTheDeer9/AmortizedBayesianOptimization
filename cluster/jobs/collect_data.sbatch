#!/bin/bash
#SBATCH --job-name=acbo_collect_data
#SBATCH --partition=training          # Use training partition for taught students
#SBATCH --qos=training               # Quality of service for training
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # 8 CPU cores for parallel collection
#SBATCH --mem=32G                    # 32GB RAM for data processing
#SBATCH --time=24:00:00              # 24 hour time limit
#SBATCH --output=/vol/bitbucket/%u/causal_bayes_opt/logs/collection/collect_data_%j.out
#SBATCH --error=/vol/bitbucket/%u/causal_bayes_opt/logs/collection/collect_data_%j.err

# Job information
echo "=================================================="
echo "ACBO Data Collection Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=================================================="

# Configuration - Edit these parameters as needed
DATASET_SIZE=${DATASET_SIZE:-"medium"}      # small/medium/large/xlarge
DIFFICULTY_LEVELS=${DIFFICULTY_LEVELS:-"all"}  # difficulty_1 difficulty_2 ... or all
OUTPUT_DIR=${OUTPUT_DIR:-"/vol/bitbucket/$USER/causal_bayes_opt/data/raw"}
WORKERS=${WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-100}
MIN_ACCURACY=${MIN_ACCURACY:-0.7}

# Setup environment
PROJECT_DIR="/vol/bitbucket/$USER/causal_bayes_opt"
cd "$PROJECT_DIR"

# Activate environment
echo "Activating environment..."
source activate_env.sh

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start data collection
echo "Starting data collection with configuration:"
echo "  Dataset size: $DATASET_SIZE"
echo "  Difficulty levels: $DIFFICULTY_LEVELS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Processing: Serial mode (avoiding pickle issues)"
echo "  Batch size: $BATCH_SIZE"
echo "  Min accuracy: $MIN_ACCURACY"
echo ""

# Build command
CMD="python scripts/collect_sft_dataset.py"
CMD="$CMD --size $DATASET_SIZE"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --serial"  # Use serial mode to avoid pickle issues
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --min-accuracy $MIN_ACCURACY"

# Add difficulty levels if not "all"
if [ "$DIFFICULTY_LEVELS" != "all" ]; then
    CMD="$CMD --difficulty $DIFFICULTY_LEVELS"
    CMD="$CMD --progressive"
fi

# Check for resume file
RESUME_FILE="$OUTPUT_DIR/checkpoints/latest_checkpoint.pkl"
if [ -f "$RESUME_FILE" ]; then
    echo "Found checkpoint file, resuming collection..."
    CMD="$CMD --resume $RESUME_FILE"
fi

echo "Executing: $CMD"
echo "=================================================="

# Execute collection
eval $CMD
COLLECTION_EXIT_CODE=$?

# Validate collected data if collection succeeded
if [ $COLLECTION_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Data collection completed successfully!"
    echo "Running validation..."
    
    python scripts/validate_dataset.py "$OUTPUT_DIR" --detailed --export-report --visualizations
    VALIDATION_EXIT_CODE=$?
    
    if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
        echo "✅ Data collection and validation completed successfully!"
    else
        echo "⚠️ Data collection succeeded but validation failed (exit code: $VALIDATION_EXIT_CODE)"
    fi
else
    echo "❌ Data collection failed with exit code: $COLLECTION_EXIT_CODE"
fi

# Print job statistics
echo ""
echo "=================================================="
echo "Job Statistics:"
echo "End time: $(date)"
echo "Duration: $SECONDS seconds"
echo "Exit code: $COLLECTION_EXIT_CODE"

# Check GPU utilization
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "Final GPU status:"
    nvidia-smi
fi

# Clean up temporary files if collection was successful
if [ $COLLECTION_EXIT_CODE -eq 0 ]; then
    echo "Cleaning up temporary files..."
    find "$OUTPUT_DIR" -name "*.tmp" -delete 2>/dev/null || true
fi

exit $COLLECTION_EXIT_CODE